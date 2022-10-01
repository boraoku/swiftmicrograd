import Foundation

public class Neuron {
    var w: [Value?]?//weights
    var b: Value?   //bias
    
    init(_ nin: Int) {
        self.w = (0..<nin).map { _ in Value(Double.random(in: -1.0...1.0)) }
        self.b = Value(Double.random(in: -1.0...1.0))
    }
    
    deinit {
        w?.removeAll()
        w = nil
        b = nil
        //print("Neuron dealloc")
    }

    public func feed(_ x:[Double]) -> Value? {
        let xValue: [Value?] = x.map { Value($0) }
        return feed(xValue)
    }

    public func feed(_ x:[Value?]) -> Value? {
        weak var self_ = self
        var sumproduct:Value? = Value(0.0)
        for (wi, xi) in zip(self_!.w!, x) {
            sumproduct = sumproduct! + (wi! * xi!)!
        }

        var act: Value? = sumproduct! + self_!.b!
        let out: Value? = tanh(act!)
        act = nil
        sumproduct = nil
        return out
    }

    public func parameters() -> [Value?]? {
        weak var self_ = self
        return self_!.w! + [self_!.b!]
    }
}
