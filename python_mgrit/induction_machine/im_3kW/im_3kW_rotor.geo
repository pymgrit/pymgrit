// May 2013 - Authors: J. Gyselinck, R.V. Sabariego

Geometry.AutoCoherence = 0;

RotorAngle_R = InitialRotorAngle + Pi/NbrSectTot - Pi/2; // initial rotor angle (radians)
RotorAngle_S = RotorAngle_R ;

R2 = u*92/2-AG;  // outer rotor radius
R3 = u*31.75/2;  // shaft radius
R1 = R2+AG/3;     // inner radius of moving band

// parameters for conductor and slot opening
h1  = u* 1;
h2  = u* 14.25;
d1  = u* 2;
Rsl = u* 4.26/2;

// characteristic lengths
uc = (TotalMemory <= 2048) ? u*3 : u*1.3;

pslo = uc* 0.3; // slot opening
psl  = uc* 0.6; // upper part slot
pslu = uc* 1; // lower part slot
psha = uc* 2; // shaft radius
pMB  = uc* 0.5;
p  = uc* 2;

Y1 = Sqrt(R2*R2-d1*d1/4) ;
Y2 = Sqrt(Rsl*Rsl-d1*d1/4) ;
Y3 = Sqrt(R1*R1-d1*d1/4) ;
RX = Rsl*Cos(Pi/NbrSectTot) ;
RY = Rsl*Sin(Pi/NbrSectTot) ;
RR = (h2-Rsl*(1+1/Sin(Pi/NbrSectTot)))/(1-1/Sin(Pi/NbrSectTot));
RX2 = RR*Cos(Pi/NbrSectTot) ;
RY2 = RR*Sin(Pi/NbrSectTot) ;


RotorBoundary_[] = {};
RotorPeriod_Ref_[] = {};
RotorPeriod_Dep_[] = {};
InnerMB_[] = {};

For i In {0:NbrSect-1}
  For j In {0:1}
    dP=newp ;
    Point(dP+0)  = {0,0,0,p};
    Point(dP+1)  = {d1/2,Y1,0,pslo/2};
    Point(dP+2)  = {0,R2,0,pslo};
    Point(dP+3)  = {d1/2,Y1-h1,0,pslo};
    Point(dP+4)  = {0,Y1-h1-Y2+RX,0,pslo};
    Point(dP+5)  = {0,Y1-h1-Y2,0,pslo};
    Point(dP+6)  = {RX,Y1-h1-Y2-RY,0,psl};
    Point(dP+7)  = {0,Y1-h1-Y2+Rsl-h2+RR,0,p};
    Point(dP+8)  = {RX2,Y1-h1-Y2+Rsl-h2+RR-RY2,0,pslu};
    Point(dP+9)  = {0,Y1-h1-Y2+Rsl-h2,0,pslu};
    Point(dP+10) = {R3*Sin(Pi/NbrSectTot),R3*Cos(Pi/NbrSectTot),0,psha};
    Point(dP+11) = {0,R3,0,psha};
    Point(dP+12) = {R2*Sin(Pi/NbrSectTot),R2*Cos(Pi/NbrSectTot),0,pMB};
    Point(dP+13) = {0,R1,0,pMB};
    Point(dP+14)  = {d1/2,Y3,0,pslo/2};
    Point(dP+15) = {R1*Sin(Pi/NbrSectTot),R1*Cos(Pi/NbrSectTot),0,pMB};

    For t In {dP+1:dP+15}
      Rotate {{0,0,1},{0,0,0}, RotorAngle_R + 2*Pi*i/NbrSectTot} {Point{t};}
    EndFor

    If (j==1)
      For t In {dP+0:dP+15}
        Symmetry {Cos(RotorAngle_S+2*Pi*i/NbrSectTot),Sin(RotorAngle_S+2*Pi*i/NbrSectTot),0,0} { Point{t}; }
      EndFor
    EndIf

    dR=newl-1;
    Line(dR+1) = {dP+8,dP+6};
    //Line(dR+2) = {dP+0,dP+10};
    Line(dR+3) = {dP+10,dP+12};
    Line(dR+4) = {dP+12,dP+15};
    //Line(dR+5) = {dP+0,dP+11};
    Line(dR+6) = {dP+11,dP+9};   Transfinite Line{dR+6} = 6 Using Progression 0.7;
    Line(dR+7) = {dP+9,dP+7};
    Line(dR+8) = {dP+7,dP+4};
    Line(dR+9) = {dP+4,dP+2}; Transfinite Line{dR+9} = 3;
    Line(dR+10) = {dP+2,dP+13};
    Line(dR+11) = {dP+3,dP+1}; Transfinite Line{dR+11} = 4;
    Circle(dR+12) = {dP+10,dP+0,dP+11};
    Circle(dR+13) = {dP+9,dP+7,dP+8}; Transfinite Line{dR+13} = 4;
    Circle(dR+14) = {dP+6,dP+5,dP+3};
    Circle(dR+15) = {dP+3,dP+5,dP+4};
    Circle(dR+16) = {dP+12,dP+0,dP+1};
    Circle(dR+17) = {dP+1,dP+0,dP+2};
    Circle(dR+18) = {dP+15,dP+0,dP+14};
    Circle(dR+19) = {dP+14,dP+0,dP+13};

    // physical lines
    OuterShaft_[] += dR+12;
    RotorBoundary_[] += {dR+12,dR+15,dR+13,dR+1,dR+16,dR+11,dR+14};

    sgn = (j==0)?1.:-1.;
    InnerMB_[] += {sgn*(dR+18),sgn*(dR+19)};

    If (NbrSectTot != NbrSect)
      If (i==0 && j==0)
        RotorPeriod_Ref_[] = {/*dR+2,*/dR+3,dR+4};
      EndIf
      If (i == NbrSect-1  && j==1)
        RotorPeriod_Dep_[] = {/*dR+2,*/dR+3,dR+4};
      EndIf
    EndIf

    rev = (j ? -1 : 1);

    Line Loop(newll) = {-dR-15,-dR-14,-dR-1,-dR-13,dR+7,dR+8};
    dH=news; Plane Surface(news) = -rev*{newll-1};
    RotorConductor_[] += dH;

    Line Loop(newll) = {dR+6,dR+13,dR+1,dR+14,dR+11,-dR-16,-dR-3,dR+12};
    dH=news; Plane Surface(news) = -rev*{newll-1};
    RotorIron_[] += dH;

    Line Loop(newll) = {-dR-17,-dR-11,dR+15,dR+9};
    dH=news; Plane Surface(news) = -rev*{newll-1};
    RotorSlotOpening_[] += dH;

    Line Loop(newll) = {dR+17,dR+10,-dR-19,-dR-18,-dR-4,dR+16};  // rotor airgap layer
    dH=news; Plane Surface(news) = -rev*{newll-1};
    RotorAirgapLayer_[] += dH;

  EndFor
EndFor

// Completing moving band
NN = #InnerMB_[] ;
k1 = (NbrPolesInModel==1)?NbrPolesInModel:NbrPolesInModel+1;
For k In {k1:NbrPolesTot-1}
  InnerMB_[] += Rotate {{0, 0, 1}, {0, 0, 0}, k*NbrSect*2*(Pi/NbrSectTot)} { Duplicata{ Line{InnerMB_[{0:NN-1}]};} };
EndFor



//-------------------------------------------------------------------
//Physical regions
//-------------------------------------------------------------------

Physical Surface(ROTOR_FE) = {RotorIron_[]};
Physical Surface(ROTOR_SLOTOPENING) = {RotorSlotOpening_[]};
Physical Surface(ROTOR_AIRGAP) = {RotorAirgapLayer_[]};

For i In {0:NbrSect-1}
  Physical Surface(ROTOR_BAR+1+i) = RotorConductor_[{2*i:2*i+1}];
EndFor

Color Orchid {Surface{RotorConductor_[]};}
Color SteelBlue {Surface{RotorIron_[]};}

If(Flag_OpenRotor)
  Color SkyBlue {Surface{RotorSlotOpening_[]};}
EndIf
If(!Flag_OpenRotor)
  Color SteelBlue {Surface{RotorSlotOpening_[]};}
EndIf
Color SkyBlue {Surface{RotorAirgapLayer_[]};}
//Color SkyBlue {Surface{RotorShaft_[]};}

Physical Line(SURF_INT) = {OuterShaft_[]};

Physical Line(ROTOR_BND_A0) =  {RotorPeriod_Ref_[]};
Physical Line(ROTOR_BND_A1) =  {RotorPeriod_Dep_[]};

For k In {0:NbrPolesTot/NbrPolesInModel-1}
  Physical Line(ROTOR_BND_MOVING_BAND+k) = {InnerMB_[{k*4*NbrSect:(k+1)*4*NbrSect-1}]};
EndFor

Coherence;
Geometry.AutoCoherence = 1;

//nicepos_rotor[] = {RotorBoundary_[], RotorPeriod_Ref_[], RotorPeriod_Dep_[]};
If(Flag_OpenRotor)
  nicepos_rotor[] = CombinedBoundary{Surface{RotorIron_[]};};
  nicepos_rotor[] += CombinedBoundary{Surface{RotorSlotOpening_[], RotorAirgapLayer_[]};};
EndIf
If(!Flag_OpenRotor)
  nicepos_rotor[] = CombinedBoundary{Surface{RotorIron_[],RotorSlotOpening_[]};};
  nicepos_rotor[] += CombinedBoundary{Surface{RotorAirgapLayer_[]};};
EndIf
