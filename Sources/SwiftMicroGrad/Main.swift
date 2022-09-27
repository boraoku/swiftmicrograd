import Foundation

@main
public struct SwiftMicroGrad {

    static var run01: Bool = false
    public static func main() {
        
        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------- MARK: 00 Basic NN skeleton --------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
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


        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //--------------------------------------------- MARK: 01 NN Basic derivative ------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        /*
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


        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //------------------------------------------  MARK: 02 NN with Weights and Bias----------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        /*
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
        */


        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //------------------------------------------ MARK: 03 NN backward propagation -----------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        /*
        //inputs x1, x2
        let x1 = Value (2.0, label: "x1")
        let x2 = Value (0.0, label: "x2")
        
        //weights w1, w2
        let w1 = Value(-3.0, label: "w1")
        let w2 = Value(1.0, label: "w2")

        //bias of the neuron
        let b = Value(6.8813735870195432, label: "bias")
        
        //x1*w1 + x2*w2 + b
        let x1w1 = x1*w1
        x1w1.label = "x1*w1"

        let x2w2 = x2*w2
        x2w2.label = "x2*w2"
        
        let x1w1x2w2 = x1w1 + x2w2
        x1w1x2w2.label = "x1*w1 + x2*w2"
        
        let n = x1w1x2w2 + b
        n.label = "n" 

        let o = n.tanh()
        o.label = "o"

        o.grad = 1.0 //initialize final output gradient to 1
        o.backward()
        n.backward()
        b.backward()
        x1w1x2w2.backward()
        x2w2.backward()
        x1w1.backward()
        print(o.drawDot())
        */


        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //------------------------------------- MARK: 04 NN backward propagation automated ------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        /*
        let x1 = Value (2.0, label: "x1")
        let x2 = Value (0.0, label: "x2")
        
        //weights w1, w2
        let w1 = Value(-3.0, label: "w1")
        let w2 = Value(1.0, label: "w2")

        //bias of the neuron
        let b = Value(6.8813735870195432, label: "bias")
        
        //x1*w1 + x2*w2 + b
        let x1w1 = x1*w1
        x1w1.label = "x1*w1"

        let x2w2 = x2*w2
        x2w2.label = "x2*w2"
        
        let x1w1x2w2 = x1w1 + x2w2
        x1w1x2w2.label = "x1*w1 + x2*w2"
        
        let n = x1w1x2w2 + b
        n.label = "n" 

        let o = n.tanh()
        o.label = "o"

        o.grad = 1.0 //initialize final output gradient to 1

        //topological sort
        var topo :[Value] = []

        func buildTopo(_ v:Value) {
            if !v.topoVisited {
                v.topoVisited = true
                for child in v.prev {
                    buildTopo(child)
                }
                topo.append(v)
            }
        }

        buildTopo(o)
        //print(topo)

        //backward propogation on the topological sort
        for node in topo.reversed() {
            node.backward()
        }

        print(o.drawDot())
        */


        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //------------------------------- MARK: 05 NN backward propagation inside the Value class -----------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        /*
        let x1 = Value (2.0, label: "x1")
        let x2 = Value (0.0, label: "x2")
        
        //weights w1, w2
        let w1 = Value(-3.0, label: "w1")
        let w2 = Value(1.0, label: "w2")

        //bias of the neuron
        let b = Value(6.8813735870195432, label: "bias")
        
        //x1*w1 + x2*w2 + b
        let x1w1 = x1*w1
        x1w1.label = "x1*w1"

        let x2w2 = x2*w2
        x2w2.label = "x2*w2"
        
        let x1w1x2w2 = x1w1 + x2w2
        x1w1x2w2.label = "x1*w1 + x2*w2"
        
        let n = x1w1x2w2 + b
        n.label = "n" 

        let o = n.tanh()
        o.label = "o"

        o.backward()
        print(o.drawDot())

        print("\n")

        let aa = Value(3.0, label: "aa")
        let bb = aa + aa
        bb.label = "bb"
        bb.backward()
        print(bb.drawDot())
        */


        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------- MARK: 06 NN new operations and reserve -------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        /*
        let x1 = Value (2.0, label: "x1")
        let x2 = Value (0.0, label: "x2")
        
        //weights w1, w2
        let w1 = Value(-3.0, label: "w1")
        let w2 = Value(1.0, label: "w2")

        //bias of the neuron
        let b = Value(6.8813735870195432, label: "bias")
        
        //x1*w1 + x2*w2 + b
        let x1w1 = x1*w1
        x1w1.label = "x1*w1"

        let x2w2 = x2*w2
        x2w2.label = "x2*w2"
        
        let x1w1x2w2 = x1w1 + x2w2
        x1w1x2w2.label = "x1*w1 + x2*w2"
        
        let n = x1w1x2w2 + b
        n.label = "n" 

        let e = (2.0 * n).exp()
        e.label = "e"

        let o = (e - 1.0) / (e + 1.0)
        o.label = "o"

        o.backward()
        print(o.drawDot())
        */


        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //------------------------------------------- MARK: 07 Neuron, Layer and MLP ------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        /*
        let x = [2.0, 3.0]
        let n = Neuron(2)
        print(n.feed(x).drawDot())
        print("\n")

        let m = Layer(2,3)
        print(m.feed(x))
        print("\n")
        
        let y = [2.0, 3.0, -1.0]
        let o = MLP(3, [4, 4, 1])
        let z = o.feed(y)[0]
        print(o.parameters().count)
        print(z)
        z.backward()
        print(z.drawDot())
        */


        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //------------------------------------------------ MARK: 08 NN Training -----------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        /*
        //input
        let xs = [ 
                [2.0, 3.0, -1.0],
                [3.0, -1.0, 0.5],
                [0.5, 1.0, 1.0],
                [1.0, 1.0, -1.0]
                ]
        //desired outputs
        let ys = [1.0, -1.0, -1.0, 1.0]

        let n = MLP(3, [4, 4, 1])

        for i in 0...499 {
            print("\n---------------Loop no \(i+1)")

            //Forward Pass to estimate loss
            let ydred = xs.map { n.feed($0) }
            let losses = zip(ydred,ys).map() { ($0[0] - $1)**2 }
            var loss = losses[0]
            for i in 1..<losses.count {
                loss = loss + losses[i]
            }
            print("Loss Before Gradient Descent \(loss.data)")

            //Zerograd - important!
            for p in n.parameters() {
                p.grad = 0.0
            }

            //Backward Pass
            loss.backward()
            //print("Before Example Grad \(n.layers[0].neurons[0].w[0].grad)")
            //print("Before Example Data \(n.layers[0].neurons[0].w[0].data)")

            //Gradient Descent
            for p in n.parameters() {
                p.data += -0.1 * p.grad
            }

            //print(" After Example Grad \(n.layers[0].neurons[0].w[0].grad)")
            //print(" After Example Data \(n.layers[0].neurons[0].w[0].data)")
        
            //Forward Pass to re-estimate loss
            let ydred2 = xs.map { n.feed($0) }
            let losses2 = zip(ydred2,ys).map() { ($0[0] - $1)**2 }
            var loss2 = losses2[0]
            for i in 1..<losses2.count {
                loss2 = loss2 + losses2[i]
            }
            print(" Loss After Gradient Descent \(loss2.data)")
            print("----------------End loop no \(i+1)")

        }

        print("\n---------------Final prediction")
        let ydred = xs.map { n.feed($0) }
        print("Target \(ys)")
        print("Guess  \(ydred)")
        */


        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //----------------------------------------------- MARK: 09 NN Ready to Use --------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        /*
        //input
        let xs = [ 
                [2.0, 3.0, -1.0],
                [3.0, -1.0, 0.5],
                [0.5, 1.0, 1.0],
                [1.0, 1.0, -1.0]
                ]
        //desired outputs
        let ys = [1.0, -1.0, -1.0, 1.0]

        let n = MLP(3, [4, 4, 1])

        n.train(inputs: xs, outputs: ys, loops:100000, stepForGradDescent: 0.05, lossThreshold: 10e-6, verbose: true)
        */


        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //-------------------------------------------------- MARK: 10 Locke ---------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------
        //---------------------------------------------------------------------------------------------------------------------

        let xs = [[1.1], [1.5], [3.0], [6.0]] //inputs
        let ys = [0.25, 0.40, 0.75, 1.0]      //targets
        
        print("\nTraining...\n")
        let numberOfRuns = 5
        let numberOfMids = xs.count-1
        var results = [Double](repeating: 0.0, count: numberOfRuns)
        var midPoints:[Double] = []
        for i in 0..<numberOfMids {
            midPoints.append((xs[i][0] + xs[i+1][0]) / 2.0)
        }
        var midPointsResults = [Double](repeating: 0.0, count: numberOfRuns*numberOfMids)
        let queue = OperationQueue()

        for i in 0..<numberOfRuns {
            queue.addOperation {
                print("Running MLP \(i+1)/\(numberOfRuns)")
                let n = MLP(1, [4,4,1])
                n.train(inputs: xs, outputs: ys, loops:10000, stepForGradDescent: 0.05, lossThreshold: 10e-5, verbose: false)
                results[i] = n.feed([2.0])[0].data
                for j in 0..<numberOfMids {
                    let midResult = n.feed([midPoints[j]])[0].data
                    midPointsResults[i*numberOfMids + j] = midResult
                }
            }
        }

        queue.waitUntilAllOperationsAreFinished()

        let average = results.reduce(0.0, +) / Double(numberOfRuns)
        
        var midPointsAverage:[Double] = []
        for i in 0..<numberOfMids {
            var midPointTotal: Double = 0.0
            for j in 0..<numberOfRuns {
                let midPointResult = midPointsResults[j*numberOfMids + i]
                midPointTotal += midPointResult
            }
            let midAverage = midPointTotal/Double(numberOfRuns)
            midPointsAverage.append(midAverage)
        }
        
        let resultsString = results.map { String(format:"%.4f", $0)  }.joined(separator: ", ")
        print("\nGuesses for Load levels at 2mm Creep Rate: " + resultsString)
        print("                               In Average: " + String(format:"%.4f", average))
        for i in 0..<numberOfMids {
            print("           Mid Point at " + String(format:"%.2f", midPoints[i]) + "mm Creep Rate: " + String(format:"%.4f", midPointsAverage[i]))
        }
    }
}
