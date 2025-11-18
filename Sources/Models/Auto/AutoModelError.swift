import Foundation

enum AutoModelError: Error {
    case invalidConfig(String? = nil)
    case noModelDataToLoad
}
