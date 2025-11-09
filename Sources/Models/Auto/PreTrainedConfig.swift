import Foundation
import MLX
import Version

/// Base class for all configuration classes. Handles a few parameters common to all models' configurations.
///
/// A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to
/// initialize a model does **not** load the model weights. It only affects the model's configuration.
///
/// Class attributes (overridden by derived classes):
/// - `modelType`: An identifier for the model type, serialized into the JSON file, and used to recreate the correct object.
/// - `hasNoDefaultsAtInit`: Whether the config class can be initialized without providing input arguments.
/// - `keysToIgnoreAtInference`: A list of keys to ignore by default when looking at dictionary outputs of the model during inference.
/// - `attributeMap`: A dict that maps model specific attribute names to the standardized naming of attributes.
///
/// Common attributes (present in all subclasses):
/// - `vocabSize`: The number of tokens in the vocabulary, which is also the first dimension of the embeddings matrix.
/// - `hiddenSize`: The hidden size of the model.
/// - `numAttentionHeads`: The number of attention heads used in the multi-head attention layers of the model.
/// - `numHiddenLayers`: The number of blocks in the model.
class PreTrainedConfig {
    // Model type
    var modelType: String?

    // Instance properties - set by the deriving class
    var baseConfigKey: String = ""
    var attributeMap: [String: String] = [:]

    // Common attributes
    var returnDict: Bool
    var outputHiddenStates: Bool
    var dtype: DType?
    private var _outputAttentions: Bool

    // Less common properties
    var tieWordEmbeddings: Bool
    var chunkSizeFeedForward: Int

    // Encoder-decoder model attributes
    var isEncoderDecoder: Bool
    var isDecoder: Bool
    var crossAttentionHiddenSize: Int?
    var addCrossAttention: Bool
    var tieEncoderDecoder: Bool

    // Fine-tuning task attributes
    var architectures: [String]?
    var finetuningTask: String?
    var id2label: [Int: String]?
    var label2id: [String: Int]?
    var taskSpecificParams: [String: Any]?
    var problemType: String?

    // Tokenizer attributes
    var tokenizerClass: String?
    var prefix: String?
    var bosTokenId: Int?
    var padTokenId: Int?
    var eosTokenId: Int?
    var sepTokenId: Int?
    var decoderStartTokenId: Int?

    // Internal attributes
    var nameOrPath: String
    var commitHash: String?
    var attnImplementationInternal: String?
    var transformersVersion: String?

    // Generation defaults (for backward compatibility)
    var maxLength: Int?
    var minLength: Int?
    var doSample: Bool?
    var earlyStopping: Bool?
    var numBeams: Int?
    var temperature: Double?
    var topK: Int?
    var topP: Double?
    var typicalP: Double?
    var repetitionPenalty: Double?
    var lengthPenalty: Double?
    var noRepeatNgramSize: Int?
    var encoderNoRepeatNgramSize: Int?
    var badWordsIds: [[Int]]?
    var numReturnSequences: Int?
    var outputScores: Bool?
    var returnDictInGenerate: Bool?
    var forcedBosTokenId: Int?
    var forcedEosTokenId: Int?
    var removeInvalidValues: Bool?
    var exponentialDecayLengthPenalty: [Double]?
    var suppressTokens: [Int]?
    var beginSuppressTokens: [Int]?
    var numBeamGroups: Int?
    var diversityPenalty: Double?

    var ropeParameters: RopeParameters?

    // Additional dynamic properties
    var additionalProperties: [String: Any] = [:]

    var outputAttentions: Bool {
        get { _outputAttentions }
        set {
            // If we set `outputAttentions` explicitly before the attn implementation, dispatch eager
            if newValue && attnImplementationInternal == nil {
                attnImplementationInternal = "eager"
            }

            if newValue && attnImplementationInternal != "eager" {
                ModelUtils.log("The `outputAttentions` attribute is not supported when using " +
                    "the `attnImplementation` set to \(attnImplementationInternal ?? "nil"). " +
                    "Please set it to 'eager' instead.")
            } else {
                _outputAttentions = newValue
            }
        }
    }

    var numLabels: Int {
        get { id2label?.count ?? 0 }
        set {
            if id2label == nil || numLabels != newValue {
                createIdLabelMaps(numLabels: newValue)
            }
        }
    }

    init(
        modelType: String? = nil,
        outputHiddenStates: Bool = false,
        outputAttentions: Bool = false,
        returnDict: Bool = true,
        dtype: String? = nil,
        tieWordEmbeddings: Bool = true,
        chunkSizeFeedForward: Int = 0,
        isEncoderDecoder: Bool = false,
        isDecoder: Bool = false,
        crossAttentionHiddenSize: Int? = nil,
        addCrossAttention: Bool = false,
        tieEncoderDecoder: Bool = false,
        architectures: [String]? = nil,
        finetuningTask: String? = nil,
        id2label: [Int: String]? = nil,
        label2id: [String: Int]? = nil,
        numLabels: Int? = nil,
        taskSpecificParams: [String: Any]? = nil,
        problemType: String? = nil,
        tokenizerClass: String? = nil,
        prefix: String? = nil,
        bosTokenId: Int? = nil,
        padTokenId: Int? = nil,
        eosTokenId: Int? = nil,
        sepTokenId: Int? = nil,
        decoderStartTokenId: Int? = nil,
        nameOrPath: String = "",
        ropeParameters: RopeParameters? = nil,
        additionalParams: [String: Any] = [:]
    ) {
        if let numLabels, let id2label, id2label.count != numLabels {
            ModelUtils.log("Warning: You passed `numLabels=\(numLabels)` " +
                "which is incompatible to the `id2label` map of length `\(id2label.count)`.")
        }

        self.modelType = modelType

        // Attributes common for all models
        self.returnDict = returnDict
        self.outputHiddenStates = outputHiddenStates
        self.dtype = ModelUtils.dtype(dtype)
        _outputAttentions = outputAttentions

        // Less common properties
        self.tieWordEmbeddings = tieWordEmbeddings
        self.chunkSizeFeedForward = chunkSizeFeedForward

        // Encoder-decoder models attributes
        self.isEncoderDecoder = isEncoderDecoder
        self.isDecoder = isDecoder
        self.crossAttentionHiddenSize = crossAttentionHiddenSize
        self.addCrossAttention = addCrossAttention
        self.tieEncoderDecoder = tieEncoderDecoder

        // Fine-tuning task attributes
        self.architectures = architectures
        self.finetuningTask = finetuningTask
        self.id2label = id2label
        self.label2id = label2id
        self.taskSpecificParams = taskSpecificParams
        self.problemType = problemType

        // Tokenizer attributes
        self.tokenizerClass = tokenizerClass
        self.prefix = prefix
        self.bosTokenId = bosTokenId
        self.padTokenId = padTokenId
        self.eosTokenId = eosTokenId
        self.sepTokenId = sepTokenId
        self.decoderStartTokenId = decoderStartTokenId

        // Name or path to the pretrained checkpoint
        self.nameOrPath = nameOrPath

        // Additional attributes
        additionalProperties = additionalParams

        self.ropeParameters = ropeParameters

        if self.id2label == nil {
            createIdLabelMaps(numLabels: numLabels ?? 2)
        }

        // Initialize generation defaults
        initializeGenerationDefaults()
    }

    private func createIdLabelMaps(numLabels: Int) {
        var id2labelMap: [Int: String] = [:]
        var label2idMap: [String: Int] = [:]

        for i in 0 ..< numLabels {
            let label = "LABEL_\(i)"
            id2labelMap[i] = label
            label2idMap[label] = i
        }

        id2label = id2labelMap
        label2id = label2idMap
    }

    private func initializeGenerationDefaults() {
        // Initialize generation defaults for backward compatibility
        maxLength = 20
        minLength = 0
        doSample = false
        earlyStopping = false
        numBeams = 1
        temperature = 1.0
        topK = 50
        topP = 1.0
        typicalP = 1.0
        repetitionPenalty = 1.0
        lengthPenalty = 1.0
        noRepeatNgramSize = 0
        encoderNoRepeatNgramSize = 0
        badWordsIds = nil
        numReturnSequences = 1
        outputScores = false
        returnDictInGenerate = false
        forcedBosTokenId = nil
        forcedEosTokenId = nil
        removeInvalidValues = false
        exponentialDecayLengthPenalty = nil
        suppressTokens = nil
        beginSuppressTokens = nil
        numBeamGroups = 1
        diversityPenalty = 0.0
    }

    /// Updates attributes of this class with attributes from a dictionary.
    func update(configDict: [String: Any]) {
        for (key, value) in configDict {
            additionalProperties[key] = value
        }
    }
}
