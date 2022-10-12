import XCTest
@testable import SwiftMicroGrad

final class SwiftMicroGradTests: XCTestCase {
    func testMLPParameterCount() throws {
        let o = MLP(3, [4, 4, 1])
        XCTAssertEqual(o.parameters()!.count, 41)
    }
    
    func testCurveFitting() throws {
        
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        let xs = [[1.1], [1.5], [3.0], [6.0]] //inputs
        let ys = [0.25, 0.40, 0.75, 1.0]      //targets
        
        print("\nTraining...\n")
        let numberOfRuns = 5
        let numberOfMids = xs.count-1
        var results = [Double](repeating: 0.0, count: numberOfRuns)
        var midPoints:[Double] = []
        for i in 0..<numberOfMids {
            midPoints.append((xs[i][0] + xs[i+1][0]) / 2.0)
        }
        var midPointsResults = [Double](repeating: 0.0, count: numberOfRuns*numberOfMids)
        let queue = OperationQueue()

        for i in 0..<numberOfRuns {
            queue.addOperation {
                
                autoreleasepool {
                    print("Running MLP \(i+1)/\(numberOfRuns)")
                    var n:MLP? = MLP(1, [4,4,1])
                    n!.train(inputs: xs, outputs: ys, loops:10000, stepForGradDescent: 0.05, lossThreshold: 10e-5, verbose: false)
                    results[i] = n!.feed([2.0])![0]!.data
                    for j in 0..<numberOfMids {
                        let midResult = n!.feed([midPoints[j]])![0]!.data
                        midPointsResults[i*numberOfMids + j] = midResult
                    }
                    n = nil
                }
            }
        }

        queue.waitUntilAllOperationsAreFinished()

        let average = results.reduce(0.0, +) / Double(numberOfRuns)
        
        var midPointsAverage:[Double] = []
        for i in 0..<numberOfMids {
            var midPointTotal: Double = 0.0
            for j in 0..<numberOfRuns {
                let midPointResult = midPointsResults[j*numberOfMids + i]
                midPointTotal += midPointResult
            }
            let midAverage = midPointTotal/Double(numberOfRuns)
            midPointsAverage.append(midAverage)
        }
        
        let resultsString = results.map { String(format:"%.4f", $0)  }.joined(separator: ", ")
        print("\nGuesses for Load levels at 2mm Creep Rate: " + resultsString)
        print("                               In Average: " + String(format:"%.4f", average))
        for i in 0..<numberOfMids {
            print("           Mid Point at " + String(format:"%.2f", midPoints[i]) + "mm Creep Rate: " + String(format:"%.4f", midPointsAverage[i]))
        }
        
        XCTAssertEqual(average*100, 53.0, accuracy: 1.0)
    }
}
