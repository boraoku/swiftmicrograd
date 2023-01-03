import XCTest
@testable import SwiftMicroGrad

final class SwiftMicroGradTests: XCTestCase {
    func testMLPParameterCount() throws {
        let o = MLP(3, [4, 4, 1], .tanh)
        XCTAssertEqual(o.parameters().count, 41)
    }
    
    enum TestData {
        case creepData
        case worldPopulation
    }
    
    func selectedData() -> TestData {
        //MARK: Select the test data here for curve fitting test below
        //return .worldPopulation
        return .creepData
    }

    func meanAndStd(values: [Double]) -> (mean: Double, std: Double) {
        let count = Double(values.count)
        let mean = values.reduce(0, +) / count // Calculate the sum of the squared differences from the mean
        let variance = values.map { pow($0 - mean, 2) }.reduce(0, +) / (count - 1) // Calculate the unbiased variance (The use of n − 1 instead of n in the formula for the sample variance is known as Bessel's correction https://en.wikipedia.org/wiki/Bessel%27s_correction
        let std = sqrt(variance) // Calculate the standard deviation (square root of variance)
        return (mean, std)
    }
    
    func testCurveFitting() throws {
        
        //MARK: Input and outdata for curve fitting test are defined here
        let xinput, yinput, targets: [Double]
        let checkAtIndex: Int
        let checkValue, checkAccuracy: Double
        
        switch selectedData() {
        case .creepData:
            xinput = [1.1, 1.5, 3.0, 6.0]
            yinput = [25.0, 40.0, 75.0, 100.0]
            targets = [1.0, 2.0, 4.0, 5.0, 8.0]
            checkAtIndex = 1
            checkValue = 54.0
            checkAccuracy = 5.0
            
        case .worldPopulation:
            xinput = [0.600, 1.000, 2.000, 2.500, 5.000, 7.700, 9.700, 10.900]
            yinput = [1700.000, 1803.000, 1928.000, 1950.000, 1987.000, 2019.000, 2050.000, 2100.000]
            targets = [0.1, 0.700, 1.500, 2.250, 3.0, 6.000, 8.000, 10.0, 15.0]
            checkAtIndex = 6
            checkValue = 2022.0
            checkAccuracy = 10.0
        }
        
        print("\nTraining...\n")
        let numberOfRuns = 1
        var results: [[Double]] = []

        for _ in 0 ..< targets.count {
            let innerArray = [Double](repeating: 0.0, count: numberOfRuns)
            results.append(innerArray)
        }
        let queue = OperationQueue()
        
        for run in 0..<numberOfRuns {
            queue.addOperation {
                print("Running MLP \(run+1)/\(numberOfRuns)")
                let n1:MLP = MLP(1, [yinput.count], .tanh)
                let n2:MLP = MLP(yinput.count, [1], .none)
                
                //n.train(inputs: xs, outputs: ys, loops: 50000, stepForGradDescent: 0.01, lossThreshold: 0.0001, normalise: true, verbose: true, concurrencyCount: i+1)
                let verbose = true
                let loops = 50000
                let stepForGradDescent = 0.01
                
                //normalise X and Y values using Mean and Standard Deviation
                var Xmean: Double = 0.0
                var Ymean: Double = 0.0
                var Xstd: Double = 1.0
                var Ystd: Double = 1.0
                
                (Xmean, Xstd) = self.meanAndStd(values: xinput)
                
                var xs: [[Double]] = [] //also for X values process x_values:[Double] to xs:[[Double]]
                for x in xinput {
                    xs.append([(x - Xmean) / Xstd])
                }
                
                (Ymean, Ystd) = self.meanAndStd(values: yinput)
                
                var ys: [Double] = []
                for y in yinput {
                    ys.append((y - Ymean) / Ystd)
                }
                
                var finalLoss: Double = 1000.0
                
                for i in 0...loops - 1 {
                    
                    //Forward Pass to estimate loss
                    //NN1 with ReLU
                    var ydred = xs.map { n1.feed($0) }
                    var losses = zip(ydred,ys).map() { ($0[0] - $1)**2 }
                    var loss = losses[0]
                    for i in 1..<losses.count {
                        loss = loss + losses[i]
                    }
                    //NN2
                    ydred = ydred.map { n2.feed($0) }
                    losses = zip(ydred,ys).map() { ($0[0] - $1)**2 }
                    loss = losses[0]
                    for i in 1..<losses.count {
                        loss = loss + losses[i]
                    }
                    
                    //verbose report at every 5000th loop step
                    if verbose && (i+1) % 5000 == 0 {
                        var concurrency = "---------------"
                        if numberOfRuns > 0 { concurrency = "--Thread no \(numberOfRuns)--" }
                        print("\n\(concurrency) Loop no \(i+1)/\(loops): Loss \(loss.data)")
                    }
                    
                    //Zerograd - important!
                    for p in n1.parameters() {
                        p.grad = 0.0
                    }
                    for p in n2.parameters() {
                        p.grad = 0.0
                    }
                    
                    //Backward Pass
                    loss.backward()
                    
                    //Gradient Descent
                    for p in n1.parameters() {
                        p.data += -stepForGradDescent * p.grad
                    }
                    for p in n2.parameters() {
                        p.data += -stepForGradDescent * p.grad
                    }
                    
                    //record final loss
                    if (i+1) == loops {
                        finalLoss = loss.data
                    }
                }
                
                let ydred = xs.map({
                    //self.outputFor($0)
                    
                    //normalise X value
                    let xValue = [Value( ($0[0] - Xmean) / Xstd )]
                    let resultN1 = n1.feed([xValue[0].data])
                    let resultNN = n2.feed([resultN1[0].data])[0].data
                    
                    //return denormalized value
                    return resultNN * Ystd + Ymean
                })
                
                if verbose {
                    var concurrency = "---------------"
                    if run > 0 { concurrency = "--Thread no \(run) : COMPLETED --" }
                    print("\n\(concurrency)\nFinal prediction after \(loops) loops with loss \(finalLoss)")
                    print("Training data outputs \(yinput)")
                    print("Prediction outputs    \(ydred)")
                }
                
                var j = 0
                for target in targets {
                    //results[j][i] = n.outputFor(target)
                    //normalise X value
                    let xValue = [Value( (target - Xmean) / Xstd )]
                    let resultN1 = n1.feed([xValue[0].data])
                    let resultNN = n2.feed([resultN1[0].data])[0].data
                    
                    //return denormalized value
                    results[j][run] = resultNN * Ystd + Ymean
                    
                    j += 1
                }
            }
        }

        queue.waitUntilAllOperationsAreFinished()

        var averageResults: [Double] = []
        for i in 0..<targets.count {
            averageResults.append(results[i].reduce(0.0, +) / Double(numberOfRuns))
            
            let resultsString = results[i].map { String(format:"%.4f", $0)  }.joined(separator: ", ")
            print("\nGuesses for input targets of \(String(format:"%.4f", targets[i])): " + resultsString)
            print("                               In Average: " + String(format:"%.4f", averageResults[i]))
        }
        
        print("\nFor Target Inputs \(targets.map { String(format:"%.4f", $0)  }.joined(separator: ", "))")
        print("Average Results \(averageResults.map { String(format:"%.4f", $0)  }.joined(separator: ", "))")
        
        XCTAssertEqual(averageResults[checkAtIndex], checkValue, accuracy: checkAccuracy)
    }
}
