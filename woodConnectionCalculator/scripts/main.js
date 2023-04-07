// Calculation function
function woodConnCalc()
{
    var CD, CM, Ct, D, S, bf, df, de, Ceg, Cdi, Ctn, theta;
    var Im, Is, II, IIIm, IIIs, IV, Zprime, RuDivZprime;

	// Input variable definitions
    Ru = parseFloat(document.getElementById("Ru").value);
    CD = parseFloat(document.getElementById("CD").value);
    CM = parseFloat(document.getElementById("CM").value);
    Ct = parseFloat(document.getElementById("Ct").value);
    D = parseFloat(document.getElementById("D").value);
    S = parseFloat(document.getElementById("S").value);
    bf = parseFloat(document.getElementById("bf").value);
    df = parseFloat(document.getElementById("df").value);
    de = parseFloat(document.getElementById("de").value);
    Ceg = parseFloat(document.getElementById("Ceg").value);
    Cdi = parseFloat(document.getElementById("Cdi").value);
    Ctn = parseFloat(document.getElementById("Ctn").value);
    theta = parseFloat(document.getElementById("theta").value);
  
  
 	// Calculation code
	// Load/slip modulus
	gamma=270000*D**(3/2); // lbf/in
	// C/S areas
	Am=24*bf; // in2
	t2=0.1; //in
	As=24*t2;
	ESYP = 1400; //ksi
    E=10100; //ksi
	u=1+gamma*S/2*(1/(ESYP*As)+1/(E*Am));
	m=u-(u**2-1)**(1/2);
	REA=Math.min((ESYP*As)/(E*Am),(E*Am)/(ESYP*As));
	n=1;
	Cg=Math.min((m*(1-m**(2*n)))/(n*((1+REA*m**n)*(1+m)-1+m**(2*n)))*(1+REA)/(1-m),1);
	// Geometry factor
	de_min=4*D; //in
	Cdelta=de/de_min;
	Ceg=1;
	Cdi=1;
	Ctn=1;
	// Reference lateral design values
	// Main member dowel bearing length
	lm=bf; //in
	SG = 0.55;
	// Main member dowel bearing strength
	Fem=6100*SG**1.45/(D**(1/2))/1000; //psi
	// Side member dowel bearing length
	ls=t2; //in
	// Side member dowel bearing strength
	Fes=38; //ksi
	Ftu=38; //ksi
	// Dowel bending yield strength
	Fyb=158; //ksi
	// Load angle
	Ktheta=1+0.25*(theta/90);
	Re=Fem/Fes;
	Rt=lm/ls;
	// Yield mode Im
	Rd=4*Ktheta;
	ZIm=D*lm*Fem/Rd*1000; //lbf
	// Yield mode Is
	Rd=4*Ktheta;
	ZIs=1000*D*ls*Fes/Rd; //lbf
	// Yield mode II
	Rd=3.6*Ktheta;
	k1=((Re+2*Re**2*(1+Rt+Rt**2)+Rt**2*Re**3)**(1/2)-Re*(1+Rt))/(1+Re);
	ZII=k1*D*ls*Fes/Rd*1000; //lbf
	// Yield mode IIIm
	Rd=3.2*Ktheta;
	k2=-1+(2*(1+Re)+(2*Fyb*(1+2*Re)*D**2)/(3*Fem*lm**2))**(1/2);
	ZIIIm=k2*D*lm*Fem/((1+2*Re)*Rd)*1000; //lbf
 	// Yield mode IIIs
	Rd=3.2*Ktheta;
	k3=-1+((2*(1+Re))/Re+(2*Fyb*(2+Re)*D**2)/(3*Fem*ls**2))**(1/2);
	ZIIIs=(k3*D*ls*Fem)/((2+Re)*Rd)*1000; //lbf
 	// Yield mode IV
	Rd=3.2*Ktheta;
    ZIV=D**2/Rd*((2*Fem*Fyb)/(3*(1+Re)))**(1/2)*1000; //lbf
    
	Z=Math.min(ZIm,ZIs,ZII,ZIIIm,ZIIIs,ZIV); //lbf
	
	// Capacity of connection
	Zprime=n*Z*CD*CM*Ct*Cg*Cdelta*Ceg*Cdi*Ctn;
	
	// Check
	RuDivZprime=Ru/Zprime;
	
  	// Output variable definitions
    document.getElementById("Im").value = ZIm;
    document.getElementById("Is").value = ZIs;
    document.getElementById("II").value = ZII;
    document.getElementById("IIIm").value = ZIIIm;
    document.getElementById("IIIs").value = ZIIIs;
    document.getElementById("IV").value = ZIV;
    document.getElementById("Zprime").value = Zprime;
    document.getElementById("RuDivZprime").value = RuDivZprime;
  
}

