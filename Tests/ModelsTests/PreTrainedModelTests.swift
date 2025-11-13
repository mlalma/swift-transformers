import Foundation
import Testing
import MLX

@testable import Hub
@testable import Models

@Suite("PreTrainedModel tests")
struct PreTrainedModelTests {
    private func compareModel1(testModel: TestModel1) throws {
        // Load model_state_dict.json and compare mappings
        let stateDictFile = Bundle.module.url(forResource: "model_state_dict", withExtension: "json")!
        let stateDictData = try Data(contentsOf: stateDictFile)
        let expectedStateDictJSON = try JSONSerialization.jsonObject(with: stateDictData) as! [String: Any]
        
        var expectedStateDict: [String: MLXArray] = [:]
        for (key, value) in expectedStateDictJSON {
            guard let tensorDict = value as? [String: Any],
                  let shape = tensorDict["shape"] as? [Int],
                  let data = tensorDict["data"] as? [Any] else {
                continue
            }
                        
            // Flatten multidimensional data recursively
            func flattenData(_ data: Any) -> [Any] {
                if let array = data as? [Any] {
                    return array.flatMap { flattenData($0) }
                } else {
                    return [data]
                }
            }
            
            let flattenedData = flattenData(data)
                        
            // Handle both Int and Double from JSON
            let floatData = flattenedData.compactMap { value -> Float? in
                if let doubleVal = value as? Double {
                    return Float(doubleVal)
                }
                return nil
            }
            expectedStateDict[key] = MLXArray(floatData, shape)
        }
                
        let expectedKeys = Set(expectedStateDict.keys)
        let actualKeys = Set(testModel.stateDict.keys)
        #expect(expectedKeys == actualKeys, "State dict keys should match")
        
        for key in expectedKeys {
            guard let actualArray = testModel.stateDict[key],
                  let expectedArray = expectedStateDict[key] else {
                #expect(Bool(false), "Model should contain key: \(key)")
                continue
            }
            
            #expect(actualArray.shape == expectedArray.shape,
                   "Shape mismatch for key '\(key)': expected \(expectedArray.shape), got \(actualArray.shape)")
            
            #expect(actualArray.dtype == expectedArray.dtype,
                   "Dtype mismatch for key '\(key)': expected \(expectedArray.dtype), got \(actualArray.dtype)")
            
            let isClose = MLX.allClose(actualArray, expectedArray, rtol: 1e-5, atol: 1e-8)
            #expect(isClose.item() == true,
                   "Value mismatch for key '\(key)': arrays are not close")
        }
        
        let extraKeys = actualKeys.subtracting(expectedKeys)
        #expect(extraKeys.isEmpty, "Model should not have extra keys: \(extraKeys)")
        
        let missingKeys = expectedKeys.subtracting(actualKeys)
        #expect(missingKeys.isEmpty, "Model should not be missing keys: \(missingKeys)")
    }
    
    @Test(".safetensors model parsing test from local directory")
    func parseSafetensorsModelFromLocalDirectory() async throws {
        let configFile = Bundle.module.url(forResource: "model", withExtension: "safetensors")!
        let modelFileDir = configFile.deletingLastPathComponent()
        
        ModelMapping.causalLMNames["testmodel"] = { TestModel1() }
        ModelMapping.configNames["testmodel"] = { NanoChatConfig(fromConfig: $0) }
        
        let model = await AutoModelForCausalLM.fromPretrained(modelFileDir.path(), modelArguments: ["use_safetensors": true]) as! TestModel1
        try compareModel1(testModel: model)
    }
    
    @Test(".bin model parsing test from local directory")
    func parseBinModelFromLocalDirectory() async throws {
        let configFile = Bundle.module.url(forResource: "pytorch_model", withExtension: "bin")!
        let modelFileDir = configFile.deletingLastPathComponent()
        
        ModelMapping.causalLMNames["testmodel"] = { TestModel1() }
        ModelMapping.configNames["testmodel"] = { NanoChatConfig(fromConfig: $0) }
        
        let model = await AutoModelForCausalLM.fromPretrained(modelFileDir.path(), modelArguments: [:]) as! TestModel1
        try compareModel1(testModel: model)
    }
}
