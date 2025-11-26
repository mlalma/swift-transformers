import Foundation

// x 1. Configuration loading phase
// x 2. Model class determination phase
// - 3. Model weight loading phase
// 4. Post processing

public class AutoModelForCausalLM: BaseAutoModelClass {
    override private init() {
        super.init()
    }

    public class func from(pretrained pretrainedModelNameOrPath: String, modelArguments: [String: Any]) async -> PreTrainedModel? {
        let model = AutoModelForCausalLM()
        return await model.from(pretrained: pretrainedModelNameOrPath, modelArguments: modelArguments, modelMapping: ModelMapping.causalLMNames)
    }
}
