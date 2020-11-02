function ex1()

	ex1=JuMP.Model()
	JuMP.@variable(ex1,x[1:2],start=1.0)
	JuMP.@NLobjective(ex1,Min,x[1]-x[2])
	JuMP.@constraint(ex1,1-x[2]>=0)
	ex1=MathOptNLPModel(ex1)#MathProgNLPModel(ex1)
	G=JuMP.Model()
	JuMP.@variable(G,x[1:2],start=1.0)
	JuMP.@constraint(G,x[1]>=0)
	JuMP.@NLobjective(G,Min,0.0)
	G=MathOptNLPModel(G)#MathProgNLPModel(G)
	H=JuMP.Model()
	JuMP.@variable(H,x[1:2],start=1.0)
	JuMP.@constraint(H,x[2]>=0)
	JuMP.@NLobjective(H,Min,0.0)
	H=MathOptNLPModel(H)#MathProgNLPModel(H)

 return MPCCNLPs(ex1,G = G,H = H)
end

function ex1bd()

	ex1bd=JuMP.Model()
	ux(i)=[Inf;1][i]
	JuMP.@variable(ex1bd,x[i=1:2], upper_bound=ux(i),start=1.0)
	JuMP.@NLobjective(ex1bd,Min,x[1]-x[2])
	ex1bd=MathOptNLPModel(ex1bd)#MathProgNLPModel(ex1bd)
	G=JuMP.Model()
	JuMP.@variable(G,x[1:2],start=1.0)
	JuMP.@constraint(G,x[1]>=0)
	JuMP.@NLobjective(G,Min,0.0)
	G=MathOptNLPModel(G) #MathProgNLPModel(G)
	H=JuMP.Model()
	JuMP.@variable(H,x[1:2],start=1.0)
	JuMP.@constraint(H,x[2]>=0)
	JuMP.@NLobjective(H,Min,0.0)
	H=MathOptNLPModel(H) #MathProgNLPModel(H)

 return MPCCNLPs(ex1bd,G = G,H = H)
end

function ex2()

	ex2=JuMP.Model()
	JuMP.@variable(ex2,x[1:2],start=-1.0)
	JuMP.@NLobjective(ex2,Min,0.5*((x[1]-1)^2+(x[2]-1)^2))
	ex2=MathOptNLPModel(ex2) #MathProgNLPModel(ex2)
	G=JuMP.Model()
	JuMP.@variable(G,x[1:2],start=1.0)
	JuMP.@constraint(G,x[1]>=0)
	JuMP.@NLobjective(G,Min,0.0)
	G=MathOptNLPModel(G) #MathProgNLPModel(G)
	H=JuMP.Model()
	JuMP.@variable(H,x[1:2],start=1.0)
	JuMP.@constraint(H,x[2]>=0)
	JuMP.@NLobjective(H,Min,0.0)
	H=MathOptNLPModel(H) #MathProgNLPModel(H)

 return MPCCNLPs(ex2,G = G,H = H)
end

# Exemple 3 :
# minimize x1+x2-x3
# s.t.     0<=x1 _|_ x2>=0
#          -4x1+x3<=0
#          -4x2+x3<=0
function ex3()

	ex3=JuMP.Model()
	JuMP.@variable(ex3,x[1:3],start=1.0)
	JuMP.@NLobjective(ex3,Min,x[1]+x[2]-x[3])
	JuMP.@NLconstraint(ex3,c1,-4*x[1]+x[3]<=0)
	JuMP.@NLconstraint(ex3,c2,-4*x[2]+x[3]<=0)
	ex3=MathOptNLPModel(ex3) #MathProgNLPModel(ex3)
	G=JuMP.Model()
	JuMP.@variable(G,x[1:3],start=1.0)
	JuMP.@constraint(G,x[1]>=0)
	JuMP.@NLobjective(G,Min,0.0)
	G=MathOptNLPModel(G) #MathProgNLPModel(G)
	H=JuMP.Model()
	JuMP.@variable(H,x[1:3],start=1.0)
	JuMP.@constraint(H,x[2]>=0)
	JuMP.@NLobjective(H,Min,0.0)
	H=MathOptNLPModel(H) #MathProgNLPModel(H)

 return MPCCNLPs(ex3,G = G,H = H)
end

#Bard1 (MacMPEC)
function bard1()

 	ex3=JuMP.Model()
	JuMP.@variable(ex3,x[1:5],start=1.0)
	JuMP.@NLobjective(ex3,Min,(x[1]-5)^2+(2*x[2]+1)^2)
	JuMP.@NLconstraint(ex3,2*(x[2]-1)-1.5*x[1]+x[3]-0.5*x[4]+x[5]==0)
	ex3=MathOptNLPModel(ex3) #MathProgNLPModel(ex3)
	G=JuMP.Model()
	JuMP.@variable(G,x[1:5],start=1.0)
	JuMP.@constraint(G,3*x[1]-x[2]-3>=0)
	JuMP.@constraint(G,-x[1]+0.5*x[2]+4>=0)
	JuMP.@constraint(G,-x[1]-x[2]+7>=0)
	JuMP.@NLobjective(G,Min,0.0)
	G=MathOptNLPModel(G) #MathProgNLPModel(G)
	H=JuMP.Model()
	JuMP.@variable(H,x[1:5],start=1.0)
	JuMP.@constraint(H,x[3]>=0)
	JuMP.@constraint(H,x[4]>=0)
	JuMP.@constraint(H,x[5]>=0)
	JuMP.@NLobjective(H,Min,0.0)
	H=MathOptNLPModel(H) #MathProgNLPModel(H)

 return MPCCNLPs(ex3,G = G,H = H)
end
