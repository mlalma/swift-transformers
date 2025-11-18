import Foundation
import MLX
import MLXNN

class NanoChatModel: Module {
    let embedTokens: Embedding
    let layers: [NanoChatDecoderLayer]
    let norm: NanoChatRMSNorm
    let rotaryEmbedding: NanoChatRotaryEmbedding
    let initialNorm: NanoChatRMSNorm
    let config: NanoChatConfig

    public init(fromConfig config: NanoChatConfig) throws {
        self.config = config
        self.embedTokens = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = try (0..<config.numHiddenLayers).map { layerIdx in
            guard let layer = NanoChatDecoderLayer(config: config, layerIdx: layerIdx) else {
                throw AutoModelError.invalidConfig("Invalid config for decoder layer \(layerIdx)")
            }
            return layer
        }
        self.norm = NanoChatRMSNorm(eps: Float(config.rmsNormEps))
        self.rotaryEmbedding = try NanoChatRotaryEmbedding(config: config)
        self.initialNorm = NanoChatRMSNorm(eps: Float(config.rmsNormEps))
    }

    public func callAsFunction(
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
}
