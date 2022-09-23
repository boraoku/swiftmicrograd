import Foundation

public class Neuron {
    var w: [Value]  //weights
    var b: Value    //bias
    
    init(_ nin: Int) {
        self.w = (0..<nin).map { _ in Value(Double.random(in: -1.0...1.0)) }
        self.b = Value(Double.random(in: -1.0...1.0))
    }

    public func feed(_ x:[Double]) -> Value {
        let xValue: [Value] = x.map { Value($0) }
        return self.feed(xValue)
    }

    public func feed(_ x:[Value]) -> Value {
        var sumproduct = Value(0.0)
        for (wi, xi) in zip(self.w, x) {
            sumproduct = sumproduct + wi * xi
        }

        let act = sumproduct + self.b
        let out = act.tanh()
        return out
    }

    public func parameters() -> [Value] {
        return self.w + [self.b]
    }
}
