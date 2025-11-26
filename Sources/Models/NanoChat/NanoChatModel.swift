import Foundation
import MLX
import MLXNN

final class NanoChatModel: Module {
    let embedTokens: Embedding
    let layers: [NanoChatDecoderLayer]
    let norm: NanoChatRMSNorm
    let rotaryEmbedding: NanoChatRotaryEmbedding
    let initialNorm: NanoChatRMSNorm
    let config: NanoChatConfig

    init(fromConfig config: NanoChatConfig, weights: [String: MLXArray]) throws {
        self.config = config
        
        guard let embedTokens = weights[Constants.embedTokens] else {
            throw AutoModelError.invalidConfig("No embedding tokens found in weights")
        }
        
        self.embedTokens = Embedding(weight: embedTokens)
        self.layers = try (0..<config.numHiddenLayers).map { layerIdx in
            return try NanoChatDecoderLayer(config: config, layerIdx: layerIdx, weights: weights)
        }
        self.norm = NanoChatRMSNorm(eps: Float(config.rmsNormEps))
        self.rotaryEmbedding = try NanoChatRotaryEmbedding(config: config)
        self.initialNorm = NanoChatRMSNorm(eps: Float(config.rmsNormEps))
    }

    func callAsFunction(
        inputIds: MLXArray,
        inputsEmbeds: MLXArray? = nil,
        positionIds: MLXArray,
    ) -> MLXArray {
        var hiddenStates = inputsEmbeds != nil ? inputsEmbeds! : embedTokens(inputIds)
        
        let positionEmbeddings = rotaryEmbedding(hiddenStates, positionIds: positionIds)
        // let pastSeenTokens = 0

        hiddenStates = initialNorm(hiddenStates)
 
        var currentHiddenStates = hiddenStates
        for decoderLayer in layers {
            currentHiddenStates = decoderLayer(
                currentHiddenStates,
                attentionMask: nil,
                positionEmbeddings: positionEmbeddings
            )
        }

        currentHiddenStates = norm(currentHiddenStates)

        return currentHiddenStates
    }
    
    struct Constants {
        static let embedTokens = "model.embed_tokens.weight"
    }
}
