import Foundation

extension FloatingPoint {
    func isClose(
        to other: Self,
        relTol: Self = 1e-6,
        absTol: Self = 0
    ) -> Bool {
        let diff = Swift.abs(self - other)
        return diff <= max(relTol * max(Swift.abs(self), Swift.abs(other)), absTol)
    }
}
