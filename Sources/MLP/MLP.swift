@main
public struct MLP {
    public static func main() {
        
        let a = Value(2.0, label: "a")
        let b = Value(-3.0, label: "b")
        let c = Value(10.0, label: "c")
        var e = a*b
        e.label = "e"
        var d = e+c
        d.label = "d"
        var f = Value(-2)
        f.label = "f"
        var L = d*f
        L.label = "L"
        //print(L)
        //print(L.prev[0].prev)
        //print(L.op)
        print(L.drawDot())
    }
}
