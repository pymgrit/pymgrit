// May 2013 - Authors: J. Gyselinck, R.V. Sabariego

Geometry.AutoCoherence = 0;

StatorAngle_  = Pi/NbrSectStatorTot-Pi/2; // initial stator angle (radians)
StatorAngle_S = StatorAngle_;

R2 = u*92/2;     // inner stator radius
R3 = u*150/2;      // outer stator radius
R1 = R2-AG/3;       // outer radius of moving band

// parameters for conductor and slot opening
h1  = u* 1;
h2  = u* 15.3;
d1  = u* 2.5;
Rsl = u* 6.36/2;
ss = 0.05;

RR = (h2-Rsl*(1+1/Sin(Pi/NbrSectStatorTot)))/(1-1/Sin(Pi/NbrSectStatorTot));
Y1 = Sqrt(R2*R2-d1*d1/4) ;
Y2 = Sqrt(RR*RR-d1*d1/4) ;
Y3 = Sqrt(R1*R1-d1*d1/4) ;
RX = Rsl*Cos(Pi/NbrSectStatorTot) ;
RY = Rsl*Sin(Pi/NbrSectStatorTot) ;
RX2 = RR*Cos(Pi/NbrSectStatorTot) ;
RY2 = RR*Sin(Pi/NbrSectStatorTot) ;

// characteristic lengths
uc = (TotalMemory <= 2048) ? u * 3 : u * 1.4 ;

pslo = uc* 0.3; // slot opening
psl  = uc* 0.6; // upper part slot
pslu = uc* 1; // lower part slot
pout = uc* 2; // outer radius
pMB  = uc* 0.5; // MB
p  = uc* 2; //

StatorPeriod_Ref_[] = {}; // Empty if no symmetry is considered
StatorPeriod_Dep_[] = {};

For i In {0:NbrSectStator-1}
  For j In {0:1}
    dP=newp;
    Point(dP+0)  = {0,0,0,p};
    Point(dP+1)  = {d1/2,Y1,0,pslo/2};
    Point(dP+2)  = {d1/2,Y1+h1,0,pslo};
    Point(dP+3)  = {0,Y1+h1+Y2,0,p};
    Point(dP+4)  = {0,Y1+h1+Y2-RR+h2-Rsl,0,p};
    Point(dP+5)  = {RX2,Y1+h1+Y2-RY2,0,psl};
    Point(dP+6)  = {RX,Y1+h1+Y2-RR+h2-Rsl-RY,0,pslu};
    Point(dP+7)  = {ss*RX+(1-ss)*RX2,ss*(Y1+h1+Y2-RR+h2-Rsl-RY)+(1-ss)*Y1+h1+Y2-RY2,0,pslu};
    Point(dP+8)  = {0,ss*(Y1+h1+Y2-RR+h2-Rsl-RY)+(1-ss)*Y1+h1+Y2-RY2,0,pslu};
    Point(dP+9)  = {0,Y1+h1+Y2-RR+h2,0,pslu};
    Point(dP+10) = {R3*Sin(Pi/NbrSectStatorTot),R3*Cos(Pi/NbrSectStatorTot),0,pout};
    Point(dP+11) = {0,R3,0,pout};
    Point(dP+12) = {R2*Sin(Pi/NbrSectStatorTot),R2*Cos(Pi/NbrSectStatorTot),0,pMB};
    Point(dP+13) = {R1*Sin(Pi/NbrSectStatorTot),R1*Cos(Pi/NbrSectStatorTot),0,pMB};
    Point(dP+14) = {0,R2,0,pMB/2};
    Point(dP+15) = {0,R1,0,pMB/2};
    Point(dP+16)  = {d1/2,Y3,0,pMB/1.5};


    For t In {dP+0:dP+16}
      Rotate {{0,0,1},{0,0,0}, StatorAngle_+2*Pi*i/NbrSectStatorTot} {Point{t};}
    EndFor

    If (j==1)
      For t In {dP+0:dP+16}
        Symmetry {Cos(StatorAngle_S+2*Pi*i/NbrSectStatorTot),Sin(StatorAngle_S+2*Pi*i/NbrSectStatorTot),0,0} { Point{t}; }
      EndFor
    EndIf

    dR=newl-1;
    Line(dR+1) = {dP+1,dP+2};   Transfinite Line{dR+1} = 4;
    Line(dR+2) = {dP+5,dP+7};
    Line(dR+3) = {dP+7,dP+6};   //Transfinite Line{dR+3} = 5;
    Line(dR+4) = {dP+7,dP+8};   Transfinite Line{dR+4} = 3;
    Circle(dR+5) = {dP+2,dP+3,dP+5}; Transfinite Line{dR+5} = 4;
    Circle(dR+6) = {dP+6,dP+4,dP+9}; Transfinite Line{dR+6} = 5;
    Circle(dR+7) = {dP+15,dP+0,dP+16};
    Circle(dR+8) = {dP+12,dP+0,dP+1};
    Circle(dR+9) = {dP+1,dP+0,dP+14};
    Circle(dR+10) = {dP+10,dP+0,dP+11};
    Line(dR+11) = {dP+14,dP+8};
    Line(dR+12) = {dP+8,dP+9};   //Transfinite Line{dR+12} = 5;
    Line(dR+13) = {dP+9,dP+11};
    Line(dR+14) = {dP+12,dP+10};
    Line(dR+15) = {dP+13,dP+12};
    Line(dR+16) = {dP+15,dP+14};
    Circle(dR+17) = {dP+13,dP+0,dP+16};


    OuterStator_[] += dR+10;
    StatorBoundary_[] += {dR+10,dR+6,dR+3,dR+2,dR+5,dR+4,dR+8,dR+1};

    If (j==0)
      OuterMB_[] += {dR+7,-dR-17};
    EndIf
    If (j==1)
      OuterMB_[] += {-dR-7,dR+17};
    EndIf

    If (NbrSectStatorTot != NbrSectStator)
      If (i==0 && j==0)
        StatorPeriod_Ref_[] = {dR+14,dR+15};
      EndIf
      If (i == NbrSectStator-1  && j==1)
        StatorPeriod_Dep_[] = {dR+14,dR+15};
      EndIf
    EndIf

    rev = (j ? -1 : 1);

    Line Loop(newll) = {dR+12,-dR-6,-dR-3,dR+4};
    dH = news; Plane Surface(news) = -rev*{newll-1};
    StatorConductor_[] += dH;

    Line Loop(newll) = {dR+1,dR+5,dR+2,dR+3,dR+6,dR+13,-dR-10,-dR-14,dR+8};
    dH = news; Plane Surface(news) = -rev*{newll-1};
    StatorIron_[] += dH;

    Line Loop(newll) = {dR+11,-dR-4,-dR-2,-dR-5,-dR-1,dR+9};
    dH = news; Plane Surface(news) = -rev*{newll-1};
    StatorSlotOpening_[] += dH;

    Line Loop(newll) = {-dR-16,dR+7,-dR-17,dR+15,dR+8,dR+9};
    dH = news; Plane Surface(news) = rev*{newll-1};
    StatorAirgapLayer_[] += dH;
  EndFor
EndFor


//Completing the moving band
NN = #OuterMB_[] ;
k1 = (NbrPolesInModel==1)?NbrPolesInModel:NbrPolesInModel+1;
For k In {k1:NbrPolesTot-1}
  OuterMB_[] += Rotate {{0, 0, 1}, {0, 0, 0}, -k*NbrSectStator*2*StatorAngle_} { Duplicata{ Line{OuterMB_[{0:NN-1}]};} };
EndFor


qq=3;
For f In {0:5}
  Con[]={};
  For i In {0:NbrSectStator/qq-1}
    If (Fmod(i,6) == f)
      For j In {0:qq-1}
        Con[] += StatorConductor_[{2*i*qq+2*j,2*i*qq+2*j+1}];
      EndFor
    EndIf
  EndFor
  If (#Con[] > 0)
    Physical Surface(STATOR_IND+1+f) = {Con[]};
    If (f == 0) Color Red {Surface{Con[]};}
    EndIf
    If (f == 1) Color SpringGreen {Surface{Con[]};}
    EndIf
    If (f == 2) Color Gold {Surface{Con[]};}
    EndIf
    If (f == 3) Color Pink {Surface{Con[]};}
    EndIf
    If (f == 4) Color ForestGreen {Surface{Con[]};}
    EndIf
    If (f == 5) Color PaleGoldenrod {Surface{Con[]};}
    EndIf
  EndIf
EndFor


//----------------------------------------------------------------------------------------
// Physical regions
//----------------------------------------------------------------------------------------

Physical Surface(STATOR_FE) = {StatorIron_[]};
Physical Surface(STATOR_SLOTOPENING) = {StatorSlotOpening_[]};
Physical Surface(STATOR_AIRGAP) = {StatorAirgapLayer_[]};

Color SteelBlue {Surface{StatorIron_[]};}
Color SkyBlue {Surface{StatorSlotOpening_[]};}
Color SkyBlue {Surface{StatorAirgapLayer_[]};}

Physical Line(SURF_EXT) = {OuterStator_[]};

Physical Line(STATOR_BND_A0) = {StatorPeriod_Ref_[]};
Physical Line(STATOR_BND_A1) = {StatorPeriod_Dep_[]};

For k In {0:NbrPolesTot/NbrPolesInModel-1}
  Physical Line(STATOR_BND_MOVING_BAND+k) = {OuterMB_[{k*4*NbrSectStator:(k+1)*4*NbrSectStator-1}]};
EndFor

//nicepos_stator[] = {StatorBoundary_[],StatorPeriod_Ref_[],StatorPeriod_Dep_[]};

Coherence;
Geometry.AutoCoherence = 1;

nicepos_stator[] = CombinedBoundary{Surface{StatorIron_[]};};
nicepos_stator[] += CombinedBoundary{Surface{StatorSlotOpening_[],StatorAirgapLayer_[]};};
