import Foundation

public class Layer {
    public var neurons: [Neuron]
    public var a: ActivationFunction
    
    public init(_ nin: Int, _ nout: Int, a:ActivationFunction) {
        self.a = a
        self.neurons = (0..<nout).map { _ in Neuron(nin, a:a) }
    }
    
    deinit {
        neurons.removeAll()
        //print("Layer dealloc")
    }

    public func feed(_ x:[Double]) -> [Value] {
        let xValue: [Value] = x.map { Value($0) }
        return self.feed(xValue)
    }

    public func feed(_ x:[Value]) -> [Value] {
        let outs = self.neurons.map { $0.feed(x) }
        return outs
    }

    public func parameters() -> [Value] {
        var params: [Value] = []
        for neuron in self.neurons {
            let ps = neuron.parameters()
            params += ps
        }
        return params
    }
}
