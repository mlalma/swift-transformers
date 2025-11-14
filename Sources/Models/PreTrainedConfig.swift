import Foundation
import Hub
import MLX
import Version

/// Base class for all model configurations. Handles parameters that are common in most models.
class PreTrainedConfig {
    // Model type
    var modelType: String?

    // Instance properties - set by the deriving class
    var baseConfigKey: String = ""
    var attributeMap: [String: String] = [:]
    var architectures: [String]?

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
    var transformersWeights: String?
    
    // RoPE parameters
    var partialRotaryFactor: Double?
    var headDim: Int?
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
        outputHiddenStates: Bool = Constants.outputHiddenStates,
        outputAttentions: Bool = Constants.outputAttentions,
        returnDict: Bool = Constants.returnDict,
        dtype: String? = nil,
        tieWordEmbeddings: Bool = Constants.tieWordEmbeddings,
        chunkSizeFeedForward: Int = Constants.chunkSizeFeedForward,
        isEncoderDecoder: Bool = Constants.isEncoderDecoder,
        isDecoder: Bool = Constants.isDecoder,
        crossAttentionHiddenSize: Int? = nil,
        addCrossAttention: Bool = Constants.addCrossAttention,
        tieEncoderDecoder: Bool = Constants.tieEncoderDecoder,
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
        nameOrPath: String = Constants.nameOrPath,
        ropeParameters: RopeParameters? = nil,
        transformersWeights: String? = nil,
        additionalParams: [String: Any] = [:],
        // Generation defaults
        maxLength: Int? = Constants.maxLength,
        minLength: Int? = Constants.minLength,
        doSample: Bool? = Constants.doSample,
        earlyStopping: Bool? = Constants.earlyStopping,
        numBeams: Int? = Constants.numBeams,
        temperature: Double? = Constants.temperature,
        topK: Int? = Constants.topK,
        topP: Double? = Constants.topP,
        typicalP: Double? = Constants.typicalP,
        repetitionPenalty: Double? = Constants.repetitionPenalty,
        lengthPenalty: Double? = Constants.lengthPenalty,
        noRepeatNgramSize: Int? = Constants.noRepeatNgramSize,
        encoderNoRepeatNgramSize: Int? = Constants.encoderNoRepeatNgramSize,
        badWordsIds: [[Int]]? = nil,
        numReturnSequences: Int? = Constants.numReturnSequences,
        outputScores: Bool? = Constants.outputScores,
        returnDictInGenerate: Bool? = Constants.returnDictInGenerate,
        forcedBosTokenId: Int? = nil,
        forcedEosTokenId: Int? = nil,
        removeInvalidValues: Bool? = Constants.removeInvalidValues,
        exponentialDecayLengthPenalty: [Double]? = nil,
        suppressTokens: [Int]? = nil,
        beginSuppressTokens: [Int]? = nil,
        numBeamGroups: Int? = Constants.numBeamGroups,
        diversityPenalty: Double? = Constants.diversityPenalty,
        partialRotaryFactor: Double? = nil,
        headDim: Int? = nil
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
        self.transformersWeights = transformersWeights
        
        // Additional attributes
        additionalProperties = additionalParams

        self.ropeParameters = ropeParameters

        if self.id2label == nil {
            createIdLabelMaps(numLabels: numLabels ?? Constants.numLabels)
        }

        // Initialize generation defaults
        self.maxLength = maxLength
        self.minLength = minLength
        self.doSample = doSample
        self.earlyStopping = earlyStopping
        self.numBeams = numBeams
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.typicalP = typicalP
        self.repetitionPenalty = repetitionPenalty
        self.lengthPenalty = lengthPenalty
        self.noRepeatNgramSize = noRepeatNgramSize
        self.encoderNoRepeatNgramSize = encoderNoRepeatNgramSize
        self.badWordsIds = badWordsIds
        self.numReturnSequences = numReturnSequences
        self.outputScores = outputScores
        self.returnDictInGenerate = returnDictInGenerate
        self.forcedBosTokenId = forcedBosTokenId
        self.forcedEosTokenId = forcedEosTokenId
        self.removeInvalidValues = removeInvalidValues
        self.exponentialDecayLengthPenalty = exponentialDecayLengthPenalty
        self.suppressTokens = suppressTokens
        self.beginSuppressTokens = beginSuppressTokens
        self.numBeamGroups = numBeamGroups
        self.diversityPenalty = diversityPenalty
        self.partialRotaryFactor = partialRotaryFactor
        self.headDim = headDim
    }

    init(fromConfig config: Config) {
        modelType = config[ConfigKeys.modelType, String.self]
        baseConfigKey = config[ConfigKeys.baseConfigKey, String.self] ?? Constants.baseConfigKey
        attributeMap = config[ConfigKeys.attributeMap, [String: String].self] ?? [:]
        returnDict = config[ConfigKeys.returnDict, Bool.self] ?? Constants.returnDict
        outputHiddenStates = config[ConfigKeys.outputHiddenStates, Bool.self] ?? Constants.outputHiddenStates
        dtype = ModelUtils.dtype(config[ConfigKeys.dtype, String.self])
        _outputAttentions = config[ConfigKeys.outputAttentions, Bool.self] ?? Constants.outputAttentions
        tieWordEmbeddings = config[ConfigKeys.tieWordEmbeddings, Bool.self] ?? Constants.tieWordEmbeddings
        chunkSizeFeedForward = config[ConfigKeys.chunkSizeFeedForward, Int.self] ?? Constants.chunkSizeFeedForward
        isEncoderDecoder = config[ConfigKeys.isEncoderDecoder, Bool.self] ?? Constants.isEncoderDecoder
        isDecoder = config[ConfigKeys.isDecoder, Bool.self] ?? Constants.isDecoder
        crossAttentionHiddenSize = config[ConfigKeys.crossAttentionHiddenSize, Int.self]
        addCrossAttention = config[ConfigKeys.addCrossAttention, Bool.self] ?? Constants.addCrossAttention
        tieEncoderDecoder = config[ConfigKeys.tieEncoderDecoder, Bool.self] ?? Constants.tieEncoderDecoder
        architectures = config[ConfigKeys.architectures, [String].self]
        finetuningTask = config[ConfigKeys.finetuningTask, String.self]
        id2label = config[ConfigKeys.id2label, [Int: String].self]
        label2id = config[ConfigKeys.label2id, [String: Int].self]
        taskSpecificParams = config[ConfigKeys.taskSpecificParams, [String: Any].self]
        problemType = config[ConfigKeys.problemType, String.self]
        tokenizerClass = config[ConfigKeys.tokenizerClass, String.self]
        prefix = config[ConfigKeys.prefix, String.self]
        bosTokenId = config[ConfigKeys.bosTokenId, Int.self]
        padTokenId = config[ConfigKeys.padTokenId, Int.self]
        eosTokenId = config[ConfigKeys.eosTokenId, Int.self]
        sepTokenId = config[ConfigKeys.sepTokenId, Int.self]
        decoderStartTokenId = config[ConfigKeys.decoderStartTokenId, Int.self]
        nameOrPath = config[ConfigKeys.nameOrPath, String.self] ?? Constants.nameOrPath
        commitHash = config[ConfigKeys.commitHash, String.self]
        attnImplementationInternal = config[ConfigKeys.attnImplementation, String.self]
        transformersVersion = config[ConfigKeys.transformersVersion, String.self]
        maxLength = config[ConfigKeys.maxLength, Int.self] ?? Constants.maxLength
        minLength = config[ConfigKeys.minLength, Int.self] ?? Constants.minLength
        doSample = config[ConfigKeys.doSample, Bool.self] ?? Constants.doSample
        earlyStopping = config[ConfigKeys.earlyStopping, Bool.self] ?? Constants.earlyStopping
        numBeams = config[ConfigKeys.numBeams, Int.self] ?? Constants.numBeams
        temperature = config[ConfigKeys.temperature, Double.self] ?? Constants.temperature
        topK = config[ConfigKeys.topK, Int.self] ?? Constants.topK
        topP = config[ConfigKeys.topP, Double.self] ?? Constants.topP
        typicalP = config[ConfigKeys.typicalP, Double.self] ?? Constants.typicalP
        repetitionPenalty = config[ConfigKeys.repetitionPenalty, Double.self] ?? Constants.repetitionPenalty
        lengthPenalty = config[ConfigKeys.lengthPenalty, Double.self] ?? Constants.lengthPenalty
        noRepeatNgramSize = config[ConfigKeys.noRepeatNgramSize, Int.self] ?? Constants.noRepeatNgramSize
        encoderNoRepeatNgramSize = config[ConfigKeys.encoderNoRepeatNgramSize, Int.self] ?? Constants.encoderNoRepeatNgramSize
        badWordsIds = config[ConfigKeys.badWordsIds, [[Int]].self]
        numReturnSequences = config[ConfigKeys.numReturnSequences, Int.self] ?? Constants.numReturnSequences
        outputScores = config[ConfigKeys.outputScores, Bool.self] ?? Constants.outputScores
        returnDictInGenerate = config[ConfigKeys.returnDictInGenerate, Bool.self] ?? Constants.returnDictInGenerate
        forcedBosTokenId = config[ConfigKeys.forcedBosTokenId, Int.self]
        forcedEosTokenId = config[ConfigKeys.forcedEosTokenId, Int.self]
        removeInvalidValues = config[ConfigKeys.removeInvalidValues, Bool.self] ?? Constants.removeInvalidValues
        exponentialDecayLengthPenalty = config[ConfigKeys.exponentialDecayLengthPenalty, [Double].self]
        suppressTokens = config[ConfigKeys.suppressTokens, [Int].self]
        beginSuppressTokens = config[ConfigKeys.beginSuppressTokens, [Int].self]
        numBeamGroups = config[ConfigKeys.numBeamGroups, Int.self] ?? Constants.numBeamGroups
        diversityPenalty = config[ConfigKeys.diversityPenalty, Double.self] ?? Constants.diversityPenalty
        transformersWeights = config[ConfigKeys.transformersWeights, String.self]
        partialRotaryFactor = config[ConfigKeys.partialRotaryFactor, Double.self]
        headDim = config[ConfigKeys.headDim, Int.self]

        // RopeParameters - if present in config
        if let ropeParameters = config[ConfigKeys.ropeParameters, Config.self] {
            self.ropeParameters = RopeParameters(fromConfig: ropeParameters)
        }

        // Additional properties - store any unrecognized keys
        additionalProperties = [:]

        // Create id2label and label2id maps if not present
        if id2label == nil {
            let numLabels = config[ConfigKeys.numLabels, Int.self] ?? Constants.numLabels
            createIdLabelMaps(numLabels: numLabels)
        }
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

    /// Updates attributes of this class with attributes from a dictionary.
    func update(configDict: [String: Any]) {
        for (key, value) in configDict {
            additionalProperties[key] = value
        }
    }

    /// Default values for configuration parameters
    enum Constants {
        // Common attributes
        static let returnDict = true
        static let outputHiddenStates = false
        static let outputAttentions = false

        // Less common properties
        static let tieWordEmbeddings = true
        static let chunkSizeFeedForward = 0

        // Encoder-decoder model attributes
        static let isEncoderDecoder = false
        static let isDecoder = false
        static let addCrossAttention = false
        static let tieEncoderDecoder = false

        // Generation defaults
        static let maxLength = 20
        static let minLength = 0
        static let doSample = false
        static let earlyStopping = false
        static let numBeams = 1
        static let temperature = 1.0
        static let topK = 50
        static let topP = 1.0
        static let typicalP = 1.0
        static let repetitionPenalty = 1.0
        static let lengthPenalty = 1.0
        static let noRepeatNgramSize = 0
        static let encoderNoRepeatNgramSize = 0
        static let numReturnSequences = 1
        static let outputScores = false
        static let returnDictInGenerate = false
        static let removeInvalidValues = false
        static let numBeamGroups = 1
        static let diversityPenalty = 0.0

        // Label defaults
        static let numLabels = 2

        // String defaults
        static let nameOrPath = ""
        static let baseConfigKey = ""
    }

    /// Config key names (snake_case strings used in configuration files)
    enum ConfigKeys {
        static let modelType = "model_type"
        static let baseConfigKey = "base_config_key"
        static let attributeMap = "attribute_map"
        static let returnDict = "return_dict"
        static let outputHiddenStates = "output_hidden_states"
        static let dtype = "dtype"
        static let outputAttentions = "output_attentions"
        static let tieWordEmbeddings = "tie_word_embeddings"
        static let chunkSizeFeedForward = "chunk_size_feed_forward"
        static let isEncoderDecoder = "is_encoder_decoder"
        static let isDecoder = "is_decoder"
        static let crossAttentionHiddenSize = "cross_attention_hidden_size"
        static let addCrossAttention = "add_cross_attention"
        static let tieEncoderDecoder = "tie_encoder_decoder"
        static let architectures = "architectures"
        static let finetuningTask = "finetuning_task"
        static let id2label = "id2label"
        static let label2id = "label2id"
        static let taskSpecificParams = "task_specific_params"
        static let problemType = "problem_type"
        static let tokenizerClass = "tokenizer_class"
        static let prefix = "prefix"
        static let bosTokenId = "bos_token_id"
        static let padTokenId = "pad_token_id"
        static let eosTokenId = "eos_token_id"
        static let sepTokenId = "sep_token_id"
        static let decoderStartTokenId = "decoder_start_token_id"
        static let nameOrPath = "name_or_path"
        static let commitHash = "_commit_hash"
        static let attnImplementation = "attn_implementation"
        static let transformersVersion = "transformers_version"
        static let maxLength = "max_length"
        static let minLength = "min_length"
        static let doSample = "do_sample"
        static let earlyStopping = "early_stopping"
        static let numBeams = "num_beams"
        static let temperature = "temperature"
        static let topK = "top_k"
        static let topP = "top_p"
        static let typicalP = "typical_p"
        static let repetitionPenalty = "repetition_penalty"
        static let lengthPenalty = "length_penalty"
        static let noRepeatNgramSize = "no_repeat_ngram_size"
        static let encoderNoRepeatNgramSize = "encoder_no_repeat_ngram_size"
        static let badWordsIds = "bad_words_ids"
        static let numReturnSequences = "num_return_sequences"
        static let outputScores = "output_scores"
        static let returnDictInGenerate = "return_dict_in_generate"
        static let forcedBosTokenId = "forced_bos_token_id"
        static let forcedEosTokenId = "forced_eos_token_id"
        static let removeInvalidValues = "remove_invalid_values"
        static let exponentialDecayLengthPenalty = "exponential_decay_length_penalty"
        static let suppressTokens = "suppress_tokens"
        static let beginSuppressTokens = "begin_suppress_tokens"
        static let numBeamGroups = "num_beam_groups"
        static let diversityPenalty = "diversity_penalty"
        static let ropeParameters = "rope_parameters"
        static let numLabels = "num_labels"
        static let transformersWeights = "transformers_weights"
        static let partialRotaryFactor = "partial_rotary_factor"
        static let headDim = "head_dim"
    }
}
