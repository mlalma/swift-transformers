import Foundation

// 1. Configuration loading phase
// 2. Model class determination phase
// 3. Model weight loading phase
// 4. Post processing

public class AutoModelForCausalLM: BaseAutoModelClass {
  override private init() {
    super.init()
  }
  
  static public func fromPretrained(_ pretrainedModelNameOrPath: String, modelArguments: [String: Any]) {
    let model = AutoModelForCausalLM()
    model.fromPretrained(pretrainedModelNameOrPath, modelArguments: modelArguments)
  }
}
