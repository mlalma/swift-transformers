import Foundation

public class BaseAutoModelClass {
  internal var modelMapping: [String: String]!
  
  internal init() {
    // This class can't be directly instantiated, use one of the derived classes to initialize it
  }
  
  internal func fromPretrained(pretrainedModelNameOrPath: (String, String), modelArguments: [String: Any]) {
    var configArgument = modelArguments["config"]
    
    // if let configArgument, conf
  }
}
