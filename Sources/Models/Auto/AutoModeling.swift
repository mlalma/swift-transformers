//
//  AutoModeling.swift
//  swift-transformers
//
//  Created by Lassi Maksimainen on 07.11.2025.
//

// 1. Configuration loading phase
// 2. Model class determination phase
// 3. Model weight loading phase
// 4. Post processing

public class AutoModelForCausalLM: BaseAutoModelClass {
  override private init() {
    super.init()
  }
  
  static public func fromPretrained(pretrainedModelNameOrPath: (String, String), modelArguments: [String: Any]) {
    let model = AutoModelForCausalLM()
    model.fromPretrained(pretrainedModelNameOrPath: pretrainedModelNameOrPath, modelArguments: modelArguments)
    
    
    
  }
}
