import Foundation
import Hub
import MLX
import MLXNN

final class NanoChatForCausalLM : PreTrainedModel {
    let model: NanoChatModel
    let vocabSize: Int
    let lmHead: Linear
    
    init(fromConfig config: NanoChatConfig?) throws {
        guard let config else {
            throw AutoModelError.invalidConfig("Wrong config for NanoChatForCausalLM")
        }
        
        model = try NanoChatModel(fromConfig: config)
        self.vocabSize = config.vocabSize
        self.lmHead = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }
    
    override func loadWeightsToModel(_ weights: [String: MLXArray]) {
        ModelUtils.log("Should be overridden by the derived model class")
    }

    public func callAsFunction(
        inputIds: MLXArray,
        inputsEmbeds: MLXArray? = nil,
        positionIds: MLXArray,
    ) -> MLXArray {
        return MLXArray([])
    }
}
