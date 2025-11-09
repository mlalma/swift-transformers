import Foundation
import Hub

public class BaseAutoModelClass {
    var modelMapping: [String: String]!

    init() {
        // This class can't be directly instantiated, use one of the derived classes to initialize it
    }

    func fromPretrained(_: String, modelArguments: [String: Any]) {
        var modelConfig = modelArguments["config"] as? Config

        if modelConfig == nil {}
    }
}
