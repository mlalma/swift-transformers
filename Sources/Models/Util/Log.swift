import Foundation

extension ModelUtils {
    #if DEBUG
        /// Lightweight wrapper around Swift's `print()` function that only executes in DEBUG builds.
        /// - Parameter s: The string to print to the console
        @inline(__always) static func log(_ s: String) {
            print(s)
        }
    #else
        /// No-op in RELEASE builds
        @inline(__always) static func log(_: String) {}
    #endif
}
