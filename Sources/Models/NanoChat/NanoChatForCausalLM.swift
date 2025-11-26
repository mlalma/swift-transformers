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
        for w in weights.keys.sorted() {
            print(w)
        }
        
        guard let lmHeadWeights = weights[Constants.lmHeadWeights] else {
            throw AutoModelError.noModelDataToLoad
        }
        
        lmHead = Linear(weight: lmHeadWeights)
        model = try NanoChatModel(fromConfig: config, weights: weights)
        ModelUtils.log("Should be overridden by the derived model class")
    }

    public func callAsFunction(
        inputIds: MLXArray,
        inputsEmbeds: MLXArray? = nil,
        positionIds: MLXArray,
    ) -> MLXArray {
        return MLXArray([])
    }
    
    struct Constants {
        static let lmHeadWeights = "lm_head.weight"
    }
}
