// Authors - J. Gyselinck, R.V. Sabariego (2013)
//
// Induction motor
//

Include "im_3kW_data.geo" ;

DefineConstant[
  Flag_AnalysisType = {1,  Choices{0="Static",  1="Time domain",  2="Frequency domain"},
    Name "Input/29Type of analysis", Highlight "Blue",
    Help Str["- Use 'Static' to compute static fields created in the machine",
      "- Use 'Time domain' to compute the dynamic response of the machine",
      "- Use 'Frequency domain' to compute steady-state phasors depending on the slip"]} ,
  Flag_SrcType_Stator = { 2, Choices{1="Current", 2="Voltage"},
    Name "Input/41Source type in stator", Highlight "Blue"},
  Flag_Cir_RotorCage = { 0 , Choices{0,1},
    Name "Input/40Use circuit in rotor cage", ReadOnly (Flag_SrcType_Stator==1)}
  slip = { 0, Min 0., Max 1, Step 0.05, Loop (Flag_AnalysisType == 2),
    Name "Input/30Slip", Highlight "AliceBlue", Visible (Flag_AnalysisType == 2)}
];

Flag_Cir = (Flag_SrcType_Stator==2);

// this not necessary anymore (thanks to the new behavior of Visibility)
If(Flag_AnalysisType!=2)
  UndefineConstant[ "Input/30Slip" ];
EndIf
variableFrequencyLoop = slip;

DefineConstant[
  Flag_NL = { (Flag_AnalysisType==2)?0:1, Choices{0,1},
    Name "Input/60Nonlinear BH-curve", ReadOnly (Flag_AnalysisType==2)?1:0},
  // FIXME: nonlinear law in frequency domain not yet implemented
  Flag_NL_law_Type = { 2, Choices{0="Analytical", 1="Interpolated",
      2="Analytical VH800-65D", 3="Interpolated VH800-65D"},
    Name "Input/61BH-curve", Highlight "Blue", Visible Flag_NL}
] ;

If(Flag_AnalysisType==2)
  UndefineConstant["Input/61BH-curve"];
EndIf


Group{
  DefineGroup[ Stator_Al, Stator_Cu ];
  DefineGroup[ Rotor_Al, Rotor_Cu ];
}

Group{
  Stator_Fe     = Region[STATOR_FE] ;
  Stator_Air    = Region[STATOR_SLOTOPENING] ;
  Stator_Airgap = Region[STATOR_AIRGAP] ;

  Stator_Bnd_A0 = Region[STATOR_BND_A0] ;
  Stator_Bnd_A1 = Region[STATOR_BND_A1] ;

  If(Flag_OpenRotor)
    Rotor_Fe     = Region[ROTOR_FE] ;
    Rotor_Air    = Region[ROTOR_SLOTOPENING] ;
  EndIf
  If(!Flag_OpenRotor)
    Rotor_Fe     = Region[{ROTOR_FE, ROTOR_SLOTOPENING}] ;
    Rotor_Air    = Region[{}] ;
  EndIf

  Rotor_Airgap = Region[ROTOR_AIRGAP] ;

  nbRotorBars = (Flag_Symmetry) ? NbrPolesInModel*NbrSectTot/NbrPolesTot : NbrSectTot ;
  For k In {1:nbRotorBars}
    Rotor_Bar~{k} = Region[ (ROTOR_BAR+k) ];
    Rotor_Bars += Region[ Rotor_Bar~{k} ];
  EndFor

  Rotor_Bnd_MB = Region[ROTOR_BND_MOVING_BAND] ;
  Rotor_Bnd_A0 = Region[ROTOR_BND_A0] ;
  Rotor_Bnd_A1 = Region[ROTOR_BND_A1] ;

  MovingBand_PhysicalNb = Region[MOVING_BAND] ;  // Fictitious number for moving band, not in the geo file
  Surf_Inf = Region[SURF_EXT] ;
  Surf_bn0 = Region[SURF_INT] ;
  Surf_cutA0 = Region[{STATOR_BND_A0, ROTOR_BND_A0}];
  Surf_cutA1 = Region[{STATOR_BND_A1, ROTOR_BND_A1}];

  Stator_Ind_Ap = Region[STATOR_IND_AP]; Stator_Ind_Am = Region[{}];
  Stator_Ind_Bp = Region[STATOR_IND_BP]; Stator_Ind_Bm = Region[{}];
  Stator_Ind_Cp = Region[{}]           ; Stator_Ind_Cm = Region[STATOR_IND_CM];
  If(NbrPolesInModel > 1)
    Stator_Ind_Am += Region[STATOR_IND_AM];
    Stator_Ind_Bm += Region[STATOR_IND_BM];
    Stator_Ind_Cp += Region[STATOR_IND_CP];
  EndIf

  PhaseA = Region[{Stator_Ind_Ap, Stator_Ind_Am}];
  PhaseB = Region[{Stator_Ind_Bp, Stator_Ind_Bm}];
  PhaseC = Region[{Stator_Ind_Cp, Stator_Ind_Cm}];

  // FIXME: Just one physical region for nice graph in Onelab
  PhaseA_pos = Region[Stator_Ind_Ap];
  PhaseB_pos = Region[Stator_Ind_Bp];
  PhaseC_pos = Region[Stator_Ind_Cm];

  Stator_IndsP = Region[{Stator_Ind_Ap, Stator_Ind_Bp, Stator_Ind_Cp}];
  Stator_IndsN = Region[{Stator_Ind_Am, Stator_Ind_Bm, Stator_Ind_Cm}];

  Stator_Inds = Region[{PhaseA, PhaseB, PhaseC}] ;
  Rotor_Inds  = Region[{}] ;

  StatorC  = Region[{}] ;
  StatorCC = Region[Stator_Fe] ;
  RotorC   = Region[Rotor_Bars] ;
  RotorCC  = Region[Rotor_Fe] ;

  // Moving band:  with or without symmetry, the BND line of the rotor must be complete
  Stator_Bnd_MB = Region[STATOR_BND_MOVING_BAND];
  For k In {1:NbrPolesTot/NbrPolesInModel}
    Rotor_Bnd_MB~{k} = Region[ (ROTOR_BND_MOVING_BAND+k-1) ];
    Rotor_Bnd_MB += Region[ Rotor_Bnd_MB~{k} ];
  EndFor
  For k In {2:NbrPolesTot/NbrPolesInModel}
    Rotor_Bnd_MBaux  += Region[ Rotor_Bnd_MB~{k} ] ;
  EndFor

  Dummy = Region[NICEPOS];   // boundary between different materials, used for animation
}

Function{
  NbrPolePairs = NbrPolesTot/2 ;

  Freq = 50  ;
  Period = 1/Freq ; // Fundamental period in s

  DefineConstant[
    Flag_ImposedSpeed = { 1, Choices{0="None", 1="Synchronous speed (no load)",
        2="Choose speed"}, Name "Input/30Imposed rotor speed [rpm]",
      Highlight "Blue", Visible Flag_AnalysisType!=2},
    myrpm = { rpm_nominal, Name "Input/31Speed [rpm]", Highlight "AliceBlue",
      ReadOnlyRange 1, Visible (Flag_ImposedSpeed==2 && Flag_AnalysisType!=2)},
    Tmec = { 0, Name "Input/32Mechanical torque [Nm]",
      Highlight "AliceBlue", Visible (!Flag_ImposedSpeed && Flag_AnalysisType!=2) },
    Frict = { 0, Name "Input/33Friction torque [Nm]",
      Highlight "AliceBlue", Visible (!Flag_ImposedSpeed && Flag_AnalysisType!=2) },
    timemax = {0.022, Name "Input/40Simulation time",
      Highlight "AliceBlue", Visible (Flag_AnalysisType==1)},
    dtime = {0.0001, Name "Input/41Time steps size",
      Highlight "AliceBlue", Visible (Flag_AnalysisType==1)},
//    NbrPeriod = {10, Name "Input/40Total number of periods",
//      Highlight "AliceBlue", Visible (Flag_AnalysisType==1)},
//    NbSteps = {100, Name "Input/41Number of time steps per period",
//      Highlight "AliceBlue", Visible (Flag_AnalysisType==1)}
    NbTrelax = {0, Name "Input/04Number of periods with damping",
      Highlight "AliceBlue", Visible 1}   // relaxation of applied voltage, for reducing the transient
  ];

  Trelax = NbTrelax*Period;
  Frelax[] = (!Flag_NL || Flag_AnalysisType==0 || $Time>Trelax) ? 1. :
             0.5*(1.-Cos[Pi*$Time/Trelax]) ; // smooth step function
  Printf("wqwqwwwww[] = %g ", Pi/Trelax);
  Printf("wwwwwww[] = %g ", timemax);
  Printf("wwwwwww[] = %g ", dtime);

  rpm_syn = 60*Freq/NbrPolePairs ;

  rpm = (Flag_ImposedSpeed==0) ? 0.:
        ((Flag_ImposedSpeed==1) ? rpm_syn : myrpm) ;

  //slip = (rpm_syn-rpm)/rpm_syn ; // slip = 1 ; ==> blocked rotor

  wr = (Flag_AnalysisType==2) ? (1-slip)*2*Pi*Freq/NbrPolePairs : rpm/60*2*Pi ; // angular rotor speed in rad_mec/s

  // imposed movement with fixed speed wr
  delta_theta[] = (Flag_ImposedSpeed) ? (dtime*wr) : ($Position-$PreviousPosition); // angle step (in rad)
  time0 = 0.;                 // initial time in s
  
  sigma[ Rotor_Bars ] = (Flag_AnalysisType==2 ? slip : 1.)*sigma_bars ;

  Stator_PhaseArea[] = SurfaceArea[]{STATOR_IND_AP} + SurfaceArea[]{STATOR_IND_AM};
  NbWires[]  = 2*Ns*NbrPolesInModel/NbrPolesTot; // Number of wires in series per phase
  Printf("asdfg[] = %g ", Ns);
  Printf("asdfg[] = %g ", NbrPolesInModel);
  Printf("asdfg[] = %g ", NbrPolesTot);
  Printf("asdfg[] = %g ", 2*Ns*NbrPolesInModel/NbrPolesTot);
  SurfCoil[] = Stator_PhaseArea[];

  pA =  0 ;
  pB =  4*Pi/3 ;
  pC =  2*Pi/3 ;

  DefineConstant[
    Irms = { IA, Name "Input/50Stator current (rms)",
      Highlight "AliceBlue", Visible (Flag_SrcType_Stator==1)},
    Vrms = { VA, Name "Input/50Stator voltage (rms)",
      Highlight "AliceBlue", Visible (Flag_SrcType_Stator==2)}
  ] ;
  VV = Vrms * Sqrt[2] ;
  II = Irms * Sqrt[2] ;

 Printf("qqqqq[] = %g ", VV);
 Printf("qqqqq[] = %g ", Irms);
 Printf("qqqqq[] = %g ", II);

  Friction[] = Frict ;
  Torque_mec[] = Tmec ;
  Inertia = inertia_fe ;

}

// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------


If(Flag_Cir)
  Include "im_3kW_circuit.pro" ;
EndIf
Include "machine_magstadyn_a.pro" ;
