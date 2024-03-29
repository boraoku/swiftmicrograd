import Foundation

public enum ActivationFunction {
    case none
    case tanh
    case ReLU
}

public class Neuron {
    public var w: [Value]  //weights
    public var b: Value    //bias
    public var a: ActivationFunction
    
    public init(_ nin: Int, a: ActivationFunction) {
        self.w = (0..<nin).map { _ in Value(Double.random(in: -1.0...1.0)) }
        self.b = Value(Double.random(in: -1.0...1.0))
        self.a = a
    }
    
    deinit {
        w.removeAll()
        //print("Neuron dealloc")
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
        
        switch a {
        case .none:
            return act
            
        case .ReLU:
            let out = relu(act)
            return out
            
        case .tanh:
            let out = tanh(act)
            return out
        }
    }

    public func parameters() -> [Value] {
        return self.w + [self.b]
    }
}
