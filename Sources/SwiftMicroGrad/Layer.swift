import Foundation

public class Layer {
    var neurons: [Neuron?]?
    
    init(_ nin: Int, _ nout: Int) {
        self.neurons = (0..<nout).map { _ in Neuron(nin) }
    }
    
    deinit {
        neurons?.removeAll()
        neurons = nil
        //print("Layer dealloc")
    }

    public func feed(_ x:[Double]) -> [Value?]? {
        let xValue: [Value] = x.map { Value($0) }
        return feed(xValue)
    }

    public func feed(_ x:[Value?]) -> [Value?]? {
        weak var self_ = self
        let outs:[Value?] = self_!.neurons!.map { $0!.feed(x) }
        return outs
    }

    public func parameters() -> [Value?]? {
        weak var self_ = self
        var params: [Value?] = []
        for neuron in self_!.neurons! {
            let ps:[Value?] = neuron!.parameters()!
            params = params + ps
        }
        return params
    }
}