package latencykernel

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/kernel/internal/resolve"
)

// A stated quantization that names the checkpoint's own method is an identity, not a choice.
//
// vLLM v0.32.0 reads the method a checkpoint declares (its quantization_config.quant_method),
// asks every method class whether it claims the checkpoint (override_quantization_method), and
// then requires the --quantization it was given to equal the result: a different value raises
// at startup, an equal one changes nothing (ModelConfig._verify_quantization,
// vllm/config/model.py:1370-1449 at 5df63a72). The weights are then served in the format the
// checkpoint stores, which the graph already states, so the kernel prices a stated value that
// names the checkpoint's own method exactly as it prices an unstated one.
//
// The graph does not carry the declared method (the vendor configuration is not a schema
// input), so identity is shown from what it does carry, the stored format of each node:
//
//   - mxfp8. A MiniMax-style checkpoint tags quant_method "mxfp8", which no override claims, so
//     the value stands as stated; ModelOptMxFp8Config loads it (quantization/__init__.py:
//     "mxfp8" maps to it). The graph must store its weights at FP8 with one 1-byte scale per 32
//     elements, the microscaling layout; the nodes the checkpoint leaves out (ignored_layers)
//     carry their own dtype. A graph stored at another width or scale layout (bf16, or FP8
//     with per-tensor or 128 x 128 scales) is not an MXFP8 checkpoint, and vLLM would then
//     quantize online or refuse the mismatch
//     (OnlineQuantizationConfig), a different layout and a different price, so it is refused.
//     "modelopt_mxfp8" is not accepted: vLLM reaches it only from a ModelOpt block with
//     quant_algo MXFP8 (modelopt.py:1205-1211), and for a quant_method "mxfp8" checkpoint the
//     same value raises. The graph cannot tell the two apart.
//   - mxfp4 and gpt_oss_mxfp4. GptOssMxfp4Config claims a checkpoint with quant_method "mxfp4"
//     whose model_type is gpt_oss and returns "gpt_oss_mxfp4" whatever was stated
//     (mxfp4.py:129-146), so either spelling names the same thing. The graph must be gpt-oss
//     (architecture GptOssForCausalLM) with its routed experts at MXFP4 and everything else
//     unquantized, which is what the checkpoint's modules_to_not_convert states.
//   - modelopt_mixed. ModelOptMixedPrecisionConfig claims quant_algo MIXED_PRECISION and reads
//     each layer's algorithm from quantized_layers (modelopt.py:1543-1660@v0.32.0). The identity
//     needs a graph that places those per-layer algorithms node by node. The catalog deriver
//     places one layout: MiniMax-M3's, MXFP8 on every linear and NVFP4 on the routed experts,
//     the router FP32 and the head BF16 (catalog/scripts/derive_graph.py,
//     minimax_m3_mixed_precision). The rule shows that layout from the graph: the MiniMax-M3
//     architecture, FP8 with the microscaling scale as the global width, every routed expert
//     node at NVFP4 and at least one such node. A graph with no NVFP4 experts (bf16, MXFP8
//     alone), another architecture, or any other width on a node is not that checkpoint's
//     layout, and the value is refused. The graph carries no declared method (see mxfp8Layout),
//     so it cannot tell this checkpoint from MiniMax-M3.1-Flash's, which stores the same
//     widths under quant_method mxfp8; a deployment that states the other name for either is
//     refused by vLLM at startup and is not a run the kernel would be asked to price.

// minimaxM3Architectures are the MiniMax-M3 model classes, which share one graph handler.
var minimaxM3Architectures = map[string]bool{
	"MiniMaxM3SparseForConditionalGeneration": true,
	"MiniMaxM3SparseForCausalLM":              true,
}

const gptOssArchitecture = "GptOssForCausalLM"

// nodeDType is the dtype a node's parameters are stored at: its own, else the global one.
func nodeDType(g *model.Graph, n model.Node) model.DType {
	if n.WeightDType != "" {
		return n.WeightDType
	}
	return g.Global.WeightDType
}

func graphStores(g *model.Graph, ok func(model.Node) bool) bool {
	seen := false
	for _, lk := range g.LayerKinds {
		for _, n := range lk.Nodes {
			if n.Op != model.OpGEMM && n.Op != model.OpGroupedGEMM {
				continue
			}
			seen = true
			if !ok(n) {
				return false
			}
		}
	}
	return seen
}

// mxfp8Layout reports whether the graph stores its weights at FP8 with the microscaling layout:
// one 1-byte scale per 32 elements. The nodes a checkpoint leaves unquantized (its
// ignored_layers: the head, the routers) carry their own dtype and are not asked.
//
// The graph's scale states no block shape, so a 1 x 32 MXFP8 checkpoint and a 32 x 32 FP8 one
// that the engine expands to one scale per 32 at load look alike. The only other source of that
// scale in the catalog is the DeepSeek-V4 family, whose FP8 checkpoint declares quant_method fp8
// with 32 x 32 blocks; construct resets its stated quantization before this runs, because
// DeepseekV4FP8Config claims a DeepSeek-V4 model_type whatever was stated
// (vllm/models/deepseek_v41/quant_config.py:175-192 at 5df63a72), so a stated mxfp8 there is not
// a mismatch in vLLM and never reaches this rule. The graph should still carry the declared
// quantization method and block shape, and this rule should require them; that is the deriver's,
// in the catalog layer.
func mxfp8Layout(g *model.Graph) bool {
	ws := g.Global.WeightScale
	return g.Global.WeightDType == model.DTypeFP8 && ws != nil && ws.Elements == 32 && ws.Bytes == 1
}

// gptOssLayout reports whether the graph is gpt-oss with MXFP4 routed experts and nothing else
// quantized.
func gptOssLayout(g *model.Graph) bool {
	return g.Global.Architecture == gptOssArchitecture && g.Global.WeightDType == model.DTypeBF16 &&
		graphStores(g, func(n model.Node) bool {
			if n.Op == model.OpGroupedGEMM {
				return nodeDType(g, n) == model.DTypeMXFP4
			}
			return nodeDType(g, n) == model.DTypeBF16
		})
}

// mixedPrecisionLayout reports whether the graph is MiniMax-M3 with MXFP8 linears and NVFP4
// routed experts, the layout its ModelOpt MIXED_PRECISION checkpoint stores: the global width is
// FP8 with the microscaling scale; the routed experts (a GroupedGEMM that is not a shared
// expert; the shared expert is a linear of the checkpoint) are NVFP4 on every layer that has
// them, and at least one layer does; the one node allowed another width is the router, a GEMM
// with one row per routed expert of its layer kind, which is FP32; every other weight node is
// the global width; the head is BF16, as the checkpoint leaves it out of quantized_layers. The
// nodes are told apart by what they are, never by their free-text role.
func mixedPrecisionLayout(g *model.Graph) bool {
	if !minimaxM3Architectures[g.Global.Architecture] || !mxfp8Layout(g) {
		return false
	}
	routed := 0
	for _, lk := range g.LayerKinds {
		experts := 0
		for _, n := range lk.Nodes {
			if n.Op == model.OpGroupedGEMM && !n.IsSharedExpert() && n.Aliases == "" {
				experts = n.Experts
			}
		}
		for _, n := range lk.Nodes {
			if (n.Op != model.OpGEMM && n.Op != model.OpGroupedGEMM) || n.Aliases != "" {
				continue
			}
			switch {
			case n.Op == model.OpGroupedGEMM && !n.IsSharedExpert():
				// A routed node that also carries the shared expert would price that linear at
				// NVFP4, where the checkpoint stores it MXFP8.
				if nodeDType(g, n) != model.DTypeNVFP4 || n.SharedExperts > 0 {
					return false
				}
				routed++
			case n.Op == model.OpGEMM && experts > 0 && n.N == experts:
				// One row per routed expert: the router, which the checkpoint stores FP32.
				if nodeDType(g, n) != model.DTypeFP32 {
					return false
				}
			default:
				if nodeDType(g, n) != model.DTypeFP8 {
					return false
				}
			}
		}
	}
	for _, n := range g.Head {
		if n.Op == model.OpGEMM && n.Aliases == "" && nodeDType(g, n) != model.DTypeBF16 {
			return false
		}
	}
	return routed > 0
}

// statedQuantization resolves a stated quantization against the checkpoint. It returns the
// value the weight format is resolved from ("" when the checkpoint's own format governs), and
// an override to record when the stated spelling differs from the one vLLM resolves to. A
// stated value it cannot show to be the checkpoint's own is refused when it is one of the
// values above and passed on otherwise (the weight-only integer names, fp8, nvfp4 and the like,
// which servedDType prices).
func statedQuantization(g *model.Graph, stated string) (string, *resolve.Override, error) {
	switch stated {
	case "mxfp8":
		if mxfp8Layout(g) {
			return "", nil, nil
		}
		return "", nil, fmt.Errorf("quantization %q is stated, and this checkpoint does not store MXFP8 "+
			"weights (its graph stores %s with scale layout %v, not FP8 with one scale per 32 elements): vLLM v0.32.0 would quantize it online, or refuse the "+
			"mismatch with the checkpoint's own method (vllm/config/model.py:1440-1449), and the "+
			"kernel prices neither", stated, g.Global.WeightDType, g.Global.WeightScale)
	case "modelopt_mxfp8":
		return "", nil, fmt.Errorf("quantization %q is stated: vLLM reaches it only from a ModelOpt block "+
			"with quant_algo MXFP8, and refuses it beside a checkpoint that tags quant_method "+
			"\"mxfp8\"; the graph does not say which this checkpoint is, so the kernel does not "+
			"price it. State \"mxfp8\" for a MiniMax-style MXFP8 checkpoint", stated)
	case "gpt_oss_mxfp4", "mxfp4":
		if gptOssLayout(g) {
			var o *resolve.Override
			if stated != "gpt_oss_mxfp4" {
				o = &resolve.Override{Field: "quantization", Requested: stated, Resolved: "gpt_oss_mxfp4",
					Reason: "vLLM v0.32.0 resolves a gpt-oss checkpoint's mxfp4 to gpt_oss_mxfp4 " +
						"(GptOssMxfp4Config.override_quantization_method)"}
			}
			return "", o, nil
		}
		// Anywhere else the stated value would be priced as an online MXFP4 quantization of every
		// weight, or accepted beside a checkpoint that declares another method, which vLLM refuses
		// (vllm/config/model.py:1440-1449). The graph does not carry the declared method.
		return "", nil, fmt.Errorf("quantization %q is stated, and this graph is not gpt-oss with MXFP4 "+
			"routed experts and everything else unquantized (architecture %q, weights %s): the only "+
			"MXFP4 checkpoint the kernel can show is gpt-oss's, and any other would be quantized online "+
			"or refused by vLLM as a mismatch with the method the checkpoint declares",
			stated, g.Global.Architecture, g.Global.WeightDType)
	case "modelopt_mixed":
		if mixedPrecisionLayout(g) {
			return "", nil, nil
		}
		return "", nil, fmt.Errorf("quantization %q is stated: it names a ModelOpt MIXED_PRECISION "+
			"checkpoint, whose per-layer algorithms (quantized_layers) this graph does not place "+
			"(architecture %q, weights %s): the only layout the kernel can show is MiniMax-M3's, "+
			"MXFP8 linears under NVFP4 routed experts. Needed for another checkpoint: a deriver "+
			"that places its per-module algorithms node by node, and a catalog entry for it",
			stated, g.Global.Architecture, g.Global.WeightDType)
	}
	return stated, nil, nil
}
