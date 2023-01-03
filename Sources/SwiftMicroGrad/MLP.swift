import Foundation

public class MLP {
    public var sz: [Int]
    public var layers: [Layer]
    public var activationFunction: ActivationFunction
    /* See commit notes (3/1/2023) as to why this part is commented out
    private var Xmean: Double = 0.0
    private var Ymean: Double = 0.0
    private var Xstd: Double = 1.0
    private var Ystd: Double = 1.0
    */
    public init(_ nin: Int, _ nouts: [Int], _ a: ActivationFunction) {
        self.sz = [nin] + nouts
        self.layers = []
        self.activationFunction = a
        for i in 0..<nouts.count {
            self.layers.append(Layer(self.sz[i], self.sz[i+1], a: a))
        }
    }
    
    deinit {
        layers.removeAll()
        //print("MLP dealloc")
    }
    /* See commit notes (3/1/2023) as to why this part is commented out
    public func outputFor(_ x:Double) -> Double {
        //normalise X value
        let xValue = [Value( (x - Xmean) / Xstd )]
        let resultNN = self.feed(xValue)[0].data
        //return denormalized value
        return resultNN * Ystd + Ymean
    }
    */
    public func feed(_ x:[Double]) -> [Value] {
        let xValue: [Value] = x.map { Value($0) }
        return self.feed(xValue)
    }

    public func feed(_ x:[Value]) -> [Value] {
        var outs = x
        for layer in self.layers {
            outs = layer.feed(outs)
        }
        return outs
    }

    public func parameters() -> [Value] {
        var params: [Value] = []
        for layer in self.layers {
            let ps = layer.parameters()
            params += ps
        }
        return params
    }
    
    public func meanAndStd(values: [Double]) -> (mean: Double, std: Double) {
        let count = Double(values.count)
        let mean = values.reduce(0, +) / count // Calculate the sum of the squared differences from the mean
        let variance = values.map { pow($0 - mean, 2) }.reduce(0, +) / (count - 1) // Calculate the unbiased variance (The use of n − 1 instead of n in the formula for the sample variance is known as Bessel's correction https://en.wikipedia.org/wiki/Bessel%27s_correction
        let std = sqrt(variance) // Calculate the standard deviation (square root of variance)
        return (mean, std)
    }

    /* See commit notes (3/1/2023) as to why this part is commented out
    public func train(inputs x_input:[Double], outputs y_input:[Double],
                      loops: Int = 30000, stepForGradDescent: Double = 0.01, lossThreshold: Double = 0.001, normalise: Bool = true, verbose: Bool = false, concurrencyCount: Int = -1)
    {
        
        //normalise X and Y values using Mean and Standard Deviation
        
        if normalise {
            (Xmean, Xstd) = meanAndStd(values: x_input)
        }
        var xs: [[Double]] = [] //also for X values process x_values:[Double] to xs:[[Double]]
        for x in x_input {
            xs.append([(x - Xmean) / Xstd])
        }

        if normalise {
            (Ymean, Ystd) = meanAndStd(values: y_input)
        }
        var ys: [Double] = []
        for y in y_input {
            ys.append((y - Ymean) / Ystd)
        }
        
        var finalLoss: Double = 1000.0
        var finalLoopNo = loops

        for i in 0...loops - 1 {
            
            //Forward Pass to estimate loss
            let ydred = xs.map { self.feed($0) }
            let losses = zip(ydred,ys).map() { ($0[0] - $1)**2 }
            var loss = losses[0]
            for i in 1..<losses.count {
                loss = loss + losses[i]
            }
            
            //verbose report at every 5000th loop step
            if verbose && (i+1) % 5000 == 0 {
                var concurrency = "---------------"
                if concurrencyCount > 0 { concurrency = "--Thread no \(concurrencyCount)--" }
                print("\n\(concurrency) Loop no \(i+1)/\(loops): Loss \(loss.data)")
            }
            
            //exit when lossThreshold is achieved
            if loss.data <= lossThreshold {
                finalLoopNo = i+1
                finalLoss = loss.data
                break
            }

            //Zerograd - important!
            for p in self.parameters() {
                p.grad = 0.0
            }

            //Backward Pass
            loss.backward()

            //Gradient Descent
            for p in self.parameters() {
                p.data += -stepForGradDescent * p.grad
            }

            //record final loss
            if (i+1) == loops {
                finalLoss = loss.data
            }
        }

        let ydred = x_input.map { self.outputFor($0) }

        if verbose {
            var concurrency = "---------------"
            if concurrencyCount > 0 { concurrency = "--Thread no \(concurrencyCount) : COMPLETED --" }
            print("\n\(concurrency)\nFinal prediction after \(finalLoopNo) loops with loss \(finalLoss)")
            print("Training data outputs \(y_input)")
            print("Prediction outputs    \(ydred)")
        }

    }
    */
}
