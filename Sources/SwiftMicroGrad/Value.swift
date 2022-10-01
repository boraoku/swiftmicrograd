import Foundation

private func lambda() {
    return
}

infix operator **

private func expSystem(_ input:Double) -> Double {
#if os(Linux)
    return Glibc.exp(input)
#else
    return Darwin.exp(input)
#endif
}

public func +(lhs: Value, other: Double) -> Value? {
    return lhs + Value(other)
}

public func +(other: Double, rhs: Value) -> Value? {
    return rhs + Value(other)
}

public func +(lhs: Value, other: Value) -> Value? {
    
    autoreleasepool {
        
        let out: Value? = Value(lhs.data + other.data, [lhs, other],"+")
        
        let _backward = {
            [unowned out] in
            lhs.grad += 1.0 * out!.grad
            other.grad += 1.0 * out!.grad
        }
        
        out!._backward = _backward
        return out
    }
}

public func -(lhs: Value, other: Double) -> Value? {
    return lhs + ((-1) * Value(other))!
}

public func -(other: Double, rhs: Value) -> Value? {
    return rhs + ((-1) * Value(other))!
}

public func -(lhs: Value, other: Value) -> Value? {
    return lhs + ((-1) * other)!
}

public func *(lhs: Value, other: Double) -> Value? {
    return lhs * Value(other)
}

public func *(other: Double, rhs: Value) -> Value? {
    return rhs * Value(other)
}

public func *(lhs: Value, other: Value) -> Value? {

    autoreleasepool {
        let out: Value? = Value(lhs.data * other.data, [lhs, other],"*")
        
        let _backward = {
            [unowned out] in
            lhs.grad += other.data * out!.grad
            other.grad += lhs.data * out!.grad
        }
        
        out!._backward = _backward
        return out
    }
}

public func **(lhs: Value, other: Double) -> Value? {

    autoreleasepool {
        let out: Value? = Value(pow(lhs.data, other), [lhs], String(format:"** %.4f", other))
        
        let _backward = {
            [unowned out] in
            lhs.grad += other * pow(lhs.data, (other - 1)) * out!.grad
        }
        
        out!._backward = _backward
        return out
    }
}

public func /(lhs: Value, other: Value) -> Value? {
    return lhs * (other**(-1.0))!
}

public func tanh(_ lhs: Value) -> Value? {
    
    autoreleasepool {
        
        let x: Double = lhs.data
        let t: Double = ( expSystem(2.0*x) - 1.0 ) / ( expSystem(2.0*x) + 1.0)
        
        let out: Value? = Value(t , [lhs], "α") //α stands for tanh
        
        let _backward = {
            [unowned out] in
            lhs.grad += (1 - pow(t,2.0)) * out!.grad
        }
        
        out!._backward = _backward
        return out
    }
}

public func exp(_ lhs: Value) -> Value? {
    
    autoreleasepool {
        
        let x: Double = lhs.data
        
        let out: Value? = Value(expSystem(x) , [lhs], "e") //e stands for exp
        
        let _backward = {
            [unowned out] in
            lhs.grad += out!.data * out!.grad
        }
        
        out!._backward = _backward
        return out
    }
}

public class Value: CustomStringConvertible {
    var data: Double
    var grad: Double
    var prev: [Value?]?
    var op: String
    var label: String
    var _backward: () -> () = lambda
    var topoVisited: Bool = false

    init(_ data: Double, _ children:[Value] = [], _ op:String = "", label:String = "", _ grad: Double = 0.0 ) {
        self.data = data
        self.grad = grad
        self.prev = children
        self.op = op
        self.label = label
    }
    
    deinit {
        prev?.removeAll()
        prev = nil
        //print("Value dealloc")
    }

    public func backward() {
        
        autoreleasepool {
            
            weak var self_ = self
            
            var topo :[Value?]? = []
            
            func buildTopo(_ v:Value?) {
                if !v!.topoVisited {
                    v!.topoVisited = true
                    for child in v!.prev! {
                        buildTopo(child)
                    }
                    topo!.append(v!)
                }
            }
            
            buildTopo(self_)
            
            self_!.grad = 1.0
            for node in topo!.reversed() {
                node!._backward()
            }
            
            topo?.removeAll()
            topo = nil
        }
    }

    public var description: String {
        weak var self_ = self
        return "Value(data=" + String(format:" data %.4f", self_!.data)
    }
}
