import Foundation
import Hub
import MLX
import MLXNN

final class NanoChatForCausalLM : PreTrainedModel {
    let config: NanoChatConfig
    var model: NanoChatModel?
    var lmHead: Linear?
    
    init(fromConfig config: NanoChatConfig?) throws {
        guard let config else {
            throw AutoModelError.invalidConfig("Wrong config for NanoChatForCausalLM")
        }
        
        self.config = config
    }
    
    override func loadWeightsToModel(_ weights: [String: MLXArray]) throws {
        guard let lmHeadWeights = weights[Constants.lmHeadWeights] else {
            throw AutoModelError.noModelDataToLoad
        }
        
        lmHead = Linear(weight: lmHeadWeights)
        model = try NanoChatModel(fromConfig: config, weights: weights)
    }

    public func callAsFunction(
        inputIds: MLXArray,
        inputsEmbeds: MLXArray? = nil,
        positionIds: MLXArray,
        logitsToKeep: Int? = nil
    ) -> MLXArray {
        guard let model, let lmHead else { return MLXArray([]) }
        
        let hiddenStates = model(inputIds: inputIds, inputsEmbeds: inputsEmbeds, positionIds: positionIds)
        
        // Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        let slicedHiddenStates: MLXArray
        if let logitsToKeep = logitsToKeep {
            // Slice from -logitsToKeep to end: hidden_states[:, -logitsToKeep:, :]
            let sequenceLength = hiddenStates.shape[1]
            let startIndex = max(0, sequenceLength - logitsToKeep)
            slicedHiddenStates = hiddenStates[0..., startIndex..., 0...]
        } else {
            slicedHiddenStates = hiddenStates
        }
        
        var logits = lmHead(slicedHiddenStates)
        
        // Apply final logit softcapping if configured
        if let softcapping = config.finalLogitSoftcapping {
            logits = logits / softcapping
            logits = MLX.tanh(logits)
            logits = logits * softcapping
        }
        
        // TO_DO: We should check labels and then add loss function accordingly here 
        
        return logits
    }
    
    struct Constants {
        static let lmHeadWeights = "lm_head.weight"
    }
}
