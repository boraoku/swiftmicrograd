@main
public struct MLP {

    static var run01: Bool = false
    public static func main() {
        
        /*
        //MARK: 00 Basic NN skeleton
        var a = Value(2.0, label: "a")
        var b = Value(-3.0, label: "b")
        var c = Value(10.0, label: "c")
        var e = a*b
        e.label = "e"
        var d = e+c
        d.label = "d"
        var f = Value(-2)
        f.label = "f"
        var L = d*f
        L.label = "L"
        print(L.drawDot())
        */


        /*
        //MARK: 01 Basic derivative
        let h = 0.001

        var a: Value
        if run01 { a = Value(2.0, label: "a") } else  { a = Value(2.0, label: "a") }

        var b = Value(-3.0, label: "b")
        var c = Value(10.0, label: "c")
        var e = a*b
        e.label = "e"
        var d = e+c
        d.label = "d"
        var f = Value(-2)
        f.label = "f"
        var L = d*f
        L.label = "L"
        let L1 = L.data
        
        print(L1)

        if run01 { a = Value(2.0 + h, label: "a") }
        b = Value(-3.0, label: "b")
        c = Value(10.0, label: "c")
        e = a*b
        e.label = "e"
        d = e+c
        d.label = "d"
        f = Value(-2)
        f.label = "f"
        L = d*f
        L.label = "L"
        var L2: Double
        if run01 { L2 = L.data } else { L2 = L.data + h } 

        print((L2-L1)/h)
        */

        
        //MARK: 02 NN with Weights and Bias
        //inputs x1, x2
        let x1 = Value (2.0, label: "x1")
        let x2 = Value (0.0, label: "x2")
        
        //weights w1, w2
        let w1 = Value(-3.0, label: "w1")
        let w2 = Value(1.0, label: "w2")

        //bias of the neuron
        let b = Value(6.8813735870195432, label: "bias")
        
        //x1*w1 + x2*w2 + b
        var x1w1 = x1*w1
        x1w1.label = "x1*w1"

        var x2w2 = x2*w2
        x2w2.label = "x2*w2"
        
        var x1w1x2w2 = x1w1 + x2w2
        x1w1x2w2.label = "x1*w1 + x2*w2"
        
        var n = x1w1x2w2 + b
        n.label = "n" 

        var o = n.tanh()
        o.label = "o"

        print(o.drawDot())

    }
}
