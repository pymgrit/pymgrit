/*
  This template file contains a generic getdp/onelab template for 2D models of
  rotating electrical machines.  Fields are computed with a vector potential
  formulation. Motion is solved with the moving band technique.

  The template begins with a section devoted to the declaration of all potential
  interface variables, i.e. template variables that can be accessed/redefined
  from outside the template. Interface variables are either
  - groups (declared with DefineGroup[...])
  - functions (declared with DefineFunction[...])
  - constants (declared with DefineConstant[...])

  This declaration mechanism
  - allows the user to define the quantities of interest only, the others being
  assigned a default value (zero or "empty" if no value given in the declaration
  sentence)
  - avoids irrelevant error messages being produced when referencing to the
  unused variables in the template

  Redeclarations are ignored.
*/

//-------------------------------------------------------------------------------------
// Groups:
// The groups declared in this section can be defined by the user according to
// the characteristics of the modelled machine in a "model.pro" file including
// this template file.

Group {
  DefineGroup[
    Stator_Inds, StatorC, StatorCC, Stator_Air, Stator_Airgap, Stator_Magnets,
    Rotor_Inds, RotorC, RotorCC, Rotor_Air, Rotor_Airgap, Rotor_Magnets, Rotor_Bars,
    Stator_IndsP, Stator_IndsN, Rotor_IndsP, Rotor_IndsN,
    Stator_Al, Rotor_Al, Stator_Cu, Rotor_Cu, Stator_Fe, Rotor_Fe,
    Stator_Bnd_MB, Rotor_Bnd_MBaux, Rotor_Bnd_MB,
    Surf_bn0, Surf_Inf, Point_ref,
    PhaseA, PhaseB, PhaseC, PhaseA_pos, PhaseB_pos, PhaseC_pos,
    Resistance_Cir, Inductance_Cir, Capacitance_Cir, DomainZt_Cir, DomainSource_Cir,
    Dummy
  ];
  // Exception: the group 'MovingBand_PhysicalNb' needs contain exactly one region
  // to pass a test done by the parser. It is declared with a dummy region "0".
  MovingBand_PhysicalNb = Region[0] ;
}

//-------------------------------------------------------------------------------------
// Constants:
// Interface constants are declared in this section. Constants declared with
// a DefineConstant[...] AND with a `Name' attribute are automatically added to
// the ONELAB parameter database. Any previous declaration in another file of a
// ONELAB paramater with the same name will link to the corresponding template
// parameter.

DefineConstant[
  // Should we output the local fields (a, b, j, ...)
  Flag_PrintFields = 1,
  // Symmetry
  AxialLength = 1,
  SymmetryFactor = 1,
  Flag_Symmetry = (SymmetryFactor==1) ? 0 : 1,
  Term_vxb = 0,
  // Time and frequency
  Freq = 534,
  Period = 1/Freq,
  time0 = 0,
  NbrPeriod = 1,
  timemax = NbrPeriod * Period,
  //NbSteps = 1,
  //delta_time = Period / NbSteps,
  // Material characteristics
  mur_fe = 1.0e3,
  sigma_al = 3.72e7, // conductivity of aluminum [S/m]
  sigma_cu = 5.9e7  // conductivity of copper [S/m]
  sigma_fe = 1.0e7, // conductivity of iron [S/m]
  Nb_max_iter = {30, Name "Nonlinear solver/Max. num. iterations", Visible 0},
  relaxation_factor = {1, Name "Nonlinear solver/Relaxation factor", Visible 0},
  stop_criterion = {1e-5, Name "Nonlinear solver/Stopping criterion", Visible 0},
  // Coil system
  NbrPhases = 3,
  FillFactor_Winding = 1,
  Factor_R_3DEffects = 1,
  // Electrical circuit
  Flag_Cir = 0,
  Flag_ParkTransformation = 0,
  Flag_ConstantSource = 0,
  Flag_SrcType_Stator = 0,
  Flag_SrcType_Rotor = 0, Flag_Cir_RotorCage = 0, Flag_Cir_StatorRings=0,
  Flag_ImposedCurrentDensity = 0,
  II = 0, VV = 0, wr = 0,
  Ia = II,
  Ib = II,
  Ic = II,
  Va = VV,
  Vb = VV,
  Vc = VV,
  pA = 0,
  pB = -2*Pi/3,
  pC = -4*Pi/3,
  Ie = 0, ID = 0, IQ = 0, I0 = 0,
  variableFrequencyLoop = wr,
  initialFrequencyLoop = 0,
  Flag_SrcType_StatorA = Flag_SrcType_Stator,
  Flag_SrcType_StatorB = Flag_SrcType_Stator,
  Flag_SrcType_StatorC = Flag_SrcType_Stator,
  // Mechanical equation
  Inertia = 0,
  // Simulation parameters
  Flag_SaveAllSteps = {0, Name "Input/00Save all time steps", Choices {0,1}},
  Clean_Results = {1, Name "Input/01Remove previous result files", Choices {0,1}},
  Flag_PWM = {0, Name "Input/02PWM input (otherwise sine)", Choices {0,1}},
  FreqPWM = {20000, Name "Input/03PWM switching frequency [Hz]", Highlight "AliceBlue", Visible Flag_PWM},
  modulationFactor = { 0.8, Name "Input/05Modulation factor", Highlight "AliceBlue", Visible Flag_PWM},
  ResId = "", // e.g. "cpu0/", ""cpu1/", ...
  ResDir = StrCat["res/", ResId],
  ExtGmsh = ".pos",
  ExtGnuplot = ".dat"
  R_ = {"Analysis", Name "GetDP/1ResolutionChoices", Visible 0},
  C_ = {"-setup_and_solve -v 3 -v2", Name "GetDP/9ComputeCommand", Visible 0},
  P_ = {"", Name "GetDP/2PostOperationChoices", Visible 0}
  wq = 0
];

//-------------------------------------------------------------------------------------
// Function declaration:
// Functions that can be defined explicitly outside the template in a calling .pro file
// are declared here so that their identifiers exist and can be referred to
// (but cannot be used) in other objects.

Function{
  DefineFunction[
    br, js,
    Resistance, Inductance, Capacitance, NbWires, SurfCoil,
    Theta_Park, Theta_Park_deg,
    RotorPosition, RotorPosition_deg, Friction, Torque_mec
  ];
}

// End of declarations
//-------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------
// Defninition of "template groups"
// Template groups represent the regions in terms of which the assembly or not
// of FE terms and the postprocessing calculations are controlled.  Template
// groups are defined in this section in terms of the above declared "user
// defined groups".

Group {
  //   DomainC    : with massive conductors
  //   DomainCC   : non-conducting domain
  //   DomainM    : with permanent magnets
  //   DomainB    : with inductors
  //   DomainS    : with imposed current density
  //   DomainL    : with linear permeability (no jacobian matrix)
  //   DomainNL   : with nonlinear permeability (jacobian matrix)
  //   DomainV    : with additional vxb term
  //   DomainKin  : region number for mechanical equation
  //   DomainDummy : region number for postpro with functions
  //   DomainPlotMovingGeo : region with boundary lines of geo for visualisation of moving rotor

  Inds = Region[ {Stator_Inds, Rotor_Inds} ] ;

  DomainM = Region[ {Stator_Magnets, Rotor_Magnets} ] ;
  If(!Flag_ImposedCurrentDensity)
    DomainB = Region[ {Inds} ] ;
    DomainS = Region[ {} ];
    Else
    DomainB = Region[ {} ] ;
    DomainS = Region[ {Inds} ];
  EndIf
  DomainPlotMovingGeo = Region[{Dummy}];

  Stator  = Region[{ StatorC, StatorCC }] ;
  Rotor   = Region[{ RotorC,  RotorCC }] ;

  Rotor_Moving = Region[{ Rotor, Rotor_Air, Rotor_Airgap, Rotor_Inds, Rotor_Bnd_MBaux} ] ;
  // used in ChangeOfCoordinates

  MB  = MovingBand2D[ MovingBand_PhysicalNb, Stator_Bnd_MB, Rotor_Bnd_MB, SymmetryFactor] ;
  Air = Region[{ Rotor_Air, Rotor_Airgap, Stator_Air, Stator_Airgap, MB } ] ;

  DomainCC = Region[{ Air, Inds, StatorCC, RotorCC }];
  DomainC  = Region[{ StatorC, RotorC }];
  Domain  = Region[{ DomainCC, DomainC }] ;

  DomainNL = Region[{}];
  If(Flag_NL)
    DomainNL = Region[ {Stator_Fe, Rotor_Fe } ];
    DomainL  = Region[ {Domain,-DomainNL} ];
  EndIf

  DomainV = Region[{}];
  If(Term_vxb) // or not dynamics in time domain + mechanics
    DomainV = Region[{ RotorC }];
  EndIf

  DomainKin = Region[1234] ; // Dummy region number for mechanical equation
  DomainDummy = Region[12345] ; // Dummy region number for postpro with functions
}



//-------------------------------------------------------------------------------------
// Defninition of functions and variables only used internally

Include "BH.pro"; // nonlinear BH caracteristic of magnetic material

Function {
  mu0 = 4.e-7 * Pi ;
  nu0 = 1. / mu0 ;
  nu [Region[{Air, Inds, Stator_Al, Rotor_Al, Stator_Cu, Rotor_Cu, Rotor_Magnets, Rotor_Bars}]]=nu0;

  If(!Flag_NL)
    nu [Region[{Stator_Fe, Rotor_Fe}]]  = 1 / (mur_fe * mu0) ;
    dhdb_NL [] = 0;
  EndIf
  If(Flag_NL)
    If(Flag_NL_law_Type==0)
      nu [Region[{Stator_Fe, Rotor_Fe}]] = nu_1a[$1] ;
      dhdb_NL [ DomainNL ] = dhdb_1a_NL[$1];
    EndIf
    If(Flag_NL_law_Type==1)
      nu [Region[{Stator_Fe, Rotor_Fe}]] = nu_1[$1] ;
      dhdb_NL [ DomainNL ] = dhdb_1_NL[$1];
    EndIf
    If(Flag_NL_law_Type==2)
      nu [Region[{Stator_Fe, Rotor_Fe}]] = nu_3kWa[$1] ;
      dhdb_NL [ DomainNL ] = dhdb_3kWa_NL[$1];
    EndIf
    If(Flag_NL_law_Type==3)
      nu [Region[{Stator_Fe, Rotor_Fe}]] = nu_3kW[$1] ;
      dhdb_NL [ DomainNL ] = dhdb_3kW_NL[$1];
    EndIf
  EndIf

  sigma[Region[{Rotor_Fe, Stator_Fe}]] = sigma_fe ;
  sigma[Region[{Rotor_Al, Stator_Al}]] = sigma_al ;
  sigma[Region[{Rotor_Cu, Stator_Cu}]] = sigma_cu ;
  sigma[Inds] = sigma_cu ;

  rho[] = 1/sigma[] ;

  Rb[] = Factor_R_3DEffects*AxialLength*FillFactor_Winding*NbWires[]^2/SurfCoil[]/sigma[] ;
  Resistance[Region[{Stator_Inds, Rotor_Inds}]] = Rb[] ;

  Idir[Region[{Stator_IndsP, Rotor_IndsP}]] =  1 ;
  Idir[Region[{Stator_IndsN, Rotor_IndsN}]] = -1 ;

  Idq0[] = Vector[ ID, IQ, I0 ] ;
  Pinv[] = Tensor[ Sin[Theta_Park[]],        Cos[Theta_Park[]],        1,
                   Sin[Theta_Park[]-2*Pi/3], Cos[Theta_Park[]-2*Pi/3], 1,
                   Sin[Theta_Park[]+2*Pi/3], Cos[Theta_Park[]+2*Pi/3], 1 ];

  P[] = 2/3 * Tensor[ Sin[Theta_Park[]], Sin[Theta_Park[]-2*Pi/3], Sin[Theta_Park[]+2*Pi/3],
                      Cos[Theta_Park[]], Cos[Theta_Park[]-2*Pi/3], Cos[Theta_Park[]+2*Pi/3],
                      1/2, 1/2, 1/2 ] ;

  Iabc[]     = Pinv[] * Idq0[] ;
  Flux_dq0[] = P[] * Vector[$Flux_a, $Flux_b, $Flux_c] ;

  Printf("wieso Freq[] = %g ", Freq);
  If(Flag_ParkTransformation)
    II = 1. ;
    IA[] = CompX[ Iabc[] ] ;
    IB[] = CompY[ Iabc[] ] ;
    IC[] = CompZ[ Iabc[] ] ;
  EndIf
  If(!Flag_ParkTransformation)
    If(!Flag_ConstantSource)
    // Flag_PWM = 1;
      If(!Flag_PWM) // sinusoidal excitation
        IA[] = modulationFactor * F_Sin_wt_p[]{2*Pi*Freq, pA} ;
        IB[] = modulationFactor * F_Sin_wt_p[]{2*Pi*Freq, pB} ;
        IC[] = modulationFactor * F_Sin_wt_p[]{2*Pi*Freq, pC} ;
        Printf('SINE');
      EndIf
      If(Flag_PWM) // PWM excitation
        fm = Freq;
        Printf("kooli fm[] = %g ", fm);
        fc = FreqPWM;
        Printf("kooli fc[] = %g ", fc);
        mf = fc/fm;
        Printf("kooli[] = %g ", mf);
        Pm = 1/fm;
        Pc = 1/fc;
        ns = 480 * mf; //1000 number of samples 
        ma = modulationFactor; // 0.8;
        Ac = 1;
        Am = ma * Ac;      
        
        phase_m_A = pA;
        phase_m_B = pB;
        phase_m_C = pC;
        
        FSaw_fc[] = 2*($Time * fc - Floor[ $Time * fc ])-1;
        FCarrier[] = Ac * FSaw_fc[]; // with the frequency of the carrier
        
        FV_out_A[]   = Am * F_Sin_wt_p[]{2*Pi*Freq, phase_m_A}; // modulator
        FV_out_B[]   = Am * F_Sin_wt_p[]{2*Pi*Freq, phase_m_B}; // modulator
        FV_out_C[]   = Am * F_Sin_wt_p[]{2*Pi*Freq, phase_m_C}; // modulator
		
        If ( ma <= 1 )		
			F_PWM_A[] = Sign[FV_out_A[] - FCarrier[]];
			F_PWM_B[] = Sign[FV_out_B[] - FCarrier[]];
			F_PWM_C[] = Sign[FV_out_C[] - FCarrier[]];
        Else
			F_PWM_A[] = -1.* Sign[FV_out_A[] - FCarrier[]];
			F_PWM_B[] = -1.* Sign[FV_out_B[] - FCarrier[]];
			F_PWM_C[] = -1.* Sign[FV_out_C[] - FCarrier[]];
        EndIf
		
        IA[] = F_PWM_A[] ;
        IB[] = F_PWM_B[] ;
        IC[] = F_PWM_C[] ; 
        Printf('PWM');
      EndIf
    EndIf
    If(Flag_ConstantSource)
      IA[] = 1. ;
      IB[] = 1. ;
      IC[] = 1. ;
      Frelax[] = 1;
    EndIf
  EndIf
  Printf("kooli[] = %g ", II);
  js[PhaseA] = II * NbWires[]/SurfCoil[] * IA[] * Idir[] * Vector[0, 0, 1] ;
  js[PhaseB] = II * NbWires[]/SurfCoil[] * IB[] * Idir[] * Vector[0, 0, 1] ;
  js[PhaseC] = II * NbWires[]/SurfCoil[] * IC[] * Idir[] * Vector[0, 0, 1] ;

  Velocity[] = wr*XYZ[]/\Vector[0,0,-1] ;

  // Maxwell stress tensor
  T_max[] = ( SquDyadicProduct[$1] - SquNorm[$1] * TensorDiag[0.5, 0.5, 0.5] ) / mu0 ;
  T_max_cplx[] = Re[0.5*(TensorV[CompX[$1]*Conj[$1], CompY[$1]*Conj[$1], CompZ[$1]*Conj[$1]]
      - $1*Conj[$1] * TensorDiag[0.5, 0.5, 0.5] ) / mu0] ;
  T_max_cplx_2f[] = 0.5*(TensorV[CompX[$1]*$1, CompY[$1]*$1, CompZ[$1]*$1] -
    $1*$1 * TensorDiag[0.5, 0.5, 0.5] ) / mu0 ; // To check ????

  AngularPosition[] = (Atan2[$Y,$X]#7 >= 0.)? #7 : #7+2*Pi ;

  RotatePZ[] = Rotate[ Vector[$X,$Y,$Z], 0, 0, $1 ] ; //Watch out: Do not use XYZ[]!
  // This has been fixed in getdp 25/04/2014 -- we could now remove all $X,$Y,$Z
  // calls

  // Torque computed in postprocessing
  Torque_mag[] = $Tstator ;
}


//-------------------------------------------------------------------------------------

Jacobian {
  { Name Vol; Case { { Region All ; Jacobian Vol; } } }
  { Name Sur; Case { { Region All ; Jacobian Sur; } } }
}

Integration {
  { Name I1 ; Case {
      { Type Gauss ;
        Case {
          { GeoElement Triangle   ; NumberOfPoints  6 ; }
	  { GeoElement Quadrangle ; NumberOfPoints  4 ; }
	  { GeoElement Line       ; NumberOfPoints  13 ; }
        }
      }
    }
  }
}

//-------------------------------------------------------------------------------------

Constraint {

  { Name MVP_2D ;
    Case {
      { Region Surf_Inf ; Type Assign; Value 0. ; }
      { Region Surf_bn0 ; Type Assign; Value 0. ; }

      If(Flag_Symmetry)
        { Region Surf_cutA1; SubRegion Region[{Surf_Inf,Surf_bn0, Point_ref}]; Type Link;
          RegionRef Surf_cutA0; SubRegionRef Region[{Surf_Inf,Surf_bn0, Point_ref}];
          Coefficient ((NbrPolesInModel%2)?-1:1) ;
	  Function RotatePZ[-NbrPolesInModel*2*Pi/NbrPolesTot]; }
        { Region Surf_cutA1; Type Link; RegionRef Surf_cutA0;
          Coefficient (NbrPolesInModel%2)?-1:1 ;
	  Function RotatePZ[-NbrPolesInModel*2*Pi/NbrPolesTot]; }

        //For the moving band
        For k In {1:SymmetryFactor-1}
        { Region Rotor_Bnd_MB~{k+1} ;
	  SubRegion Rotor_Bnd_MB~{(k!=SymmetryFactor-1)?k+2:1}; Type Link;
          RegionRef Rotor_Bnd_MB_1; SubRegionRef Rotor_Bnd_MB_2;
          Coefficient ((NbrPolesInModel%2)?-1:1)^(k);
	  Function RotatePZ[-k*NbrPolesInModel*2*Pi/NbrPolesTot]; }
        EndFor

      EndIf
    }
  }

  { Name Current_2D ;
    Case {
      If(Flag_SrcType_Stator==1)
        { Region PhaseA     ; Value Ia*Idir[] ; TimeFunction IA[]; }
        { Region PhaseB     ; Value Ib*Idir[] ; TimeFunction IB[]; }
        { Region PhaseC     ; Value Ic*Idir[] ; TimeFunction IC[]; }
      EndIf
      If(Flag_SrcType_Rotor==1)
        { Region Rotor_Inds ; Value Ie*Idir[] ; }
      EndIf
    }
  }

  { Name Voltage_2D ;
    Case {
      If(!Flag_Cir_RotorCage)
        { Region RotorC  ; Value 0. ; }
      EndIf
      If(!Flag_Cir_StatorRings)
        { Region StatorC ; Value 0. ; }
      EndIf
    }
  }

  { Name Current_Cir ;
    Case {
      If(Flag_Cir && Flag_SrcType_Stator==1)
        { Region Input1  ; Value Ia  ; TimeFunction IA[]; }
        { Region Input2  ; Value Ib  ; TimeFunction IB[]; }
        { Region Input3  ; Value Ic  ; TimeFunction IC[]; }
      EndIf
    }
  }

  { Name Voltage_Cir ;
    Case {
      If(Flag_Cir && Flag_SrcType_Stator==2)
        { Region Input1  ; Value Va  ; TimeFunction IA[]*Frelax[]; }
        { Region Input2  ; Value Vb  ; TimeFunction IB[]*Frelax[]; }
        { Region Input3  ; Value Vc  ; TimeFunction IC[]*Frelax[]; }
      EndIf
    }
  }


  //Kinetics
  { Name CurrentPosition ; // [m]
    Case {
      { Region DomainKin ; Type Init ; Value ($PreviousPosition = 0) ; }
    }
  }

  { Name CurrentVelocity ; // [rad/s]
    Case {
      { Region DomainKin ; Type Init ; Value 0. ; }
    }
  }

}

//-------------------------------------------------------------------------------------------

FunctionSpace {
  // Magnetic Vector Potential
  { Name Hcurl_a_2D ; Type Form1P ;
    BasisFunction {
      { Name se1 ; NameOfCoef ae1 ; Function BF_PerpendicularEdge ;
        Support Region[{ Domain, Rotor_Bnd_MBaux }] ; Entity NodesOf [ All ] ; }
   }
    Constraint {
      { NameOfCoef ae1 ; EntityType NodesOf ; NameOfConstraint MVP_2D ; }
    }
  }

  // Gradient of Electric scalar potential (2D)
  { Name Hregion_u_Mag_2D ; Type Form1P ;
    BasisFunction {
      { Name sr ; NameOfCoef ur ; Function BF_RegionZ ;
        Support DomainC ; Entity DomainC ; }
    }
    GlobalQuantity {
      { Name U ; Type AliasOf        ; NameOfCoef ur ; }
      { Name I ; Type AssociatedWith ; NameOfCoef ur ; }
    }
    Constraint {
      { NameOfCoef U ; EntityType GroupsOfNodesOf ; NameOfConstraint Voltage_2D ; }
      { NameOfCoef I ; EntityType GroupsOfNodesOf ; NameOfConstraint Current_2D ; }
    }
  }

  { Name Hregion_i_Mag_2D ; Type Vector ;
    BasisFunction {
      { Name sr ; NameOfCoef ir ; Function BF_RegionZ ;
        Support DomainB ; Entity DomainB ; }
    }
    GlobalQuantity {
      { Name Ib ; Type AliasOf        ; NameOfCoef ir ; }
      { Name Ub ; Type AssociatedWith ; NameOfCoef ir ; }
    }
    Constraint {
      { NameOfCoef Ub ; EntityType Region ; NameOfConstraint Voltage_2D ; }
      { NameOfCoef Ib ; EntityType Region ; NameOfConstraint Current_2D ; }
    }
  }

  // For circuit equations
  { Name Hregion_Z ; Type Scalar ;
    BasisFunction {
      { Name sr ; NameOfCoef ir ; Function BF_Region ;
        Support DomainZt_Cir ; Entity DomainZt_Cir ; }
    }
    GlobalQuantity {
      { Name Iz ; Type AliasOf        ; NameOfCoef ir ; }
      { Name Uz ; Type AssociatedWith ; NameOfCoef ir ; }
    }
    Constraint {
      { NameOfCoef Uz ; EntityType Region ; NameOfConstraint Voltage_Cir ; }
      { NameOfCoef Iz ; EntityType Region ; NameOfConstraint Current_Cir ; }
    }
  }

  // For mechanical equation
  { Name Position ; Type Scalar ;
    BasisFunction {
      { Name sr ; NameOfCoef pr ; Function BF_Region ;
        Support DomainKin ; Entity DomainKin ; }
    }
    GlobalQuantity {
      { Name P ; Type AliasOf  ; NameOfCoef pr ; }
    }
    Constraint {
      { NameOfCoef P ; EntityType Region ; NameOfConstraint CurrentPosition ; }
    }
  }


  { Name Velocity ; Type Scalar ;
    BasisFunction {
      { Name sr ; NameOfCoef vr ; Function BF_Region ;
        Support DomainKin ; Entity DomainKin ; } }
    GlobalQuantity {
      { Name V ; Type AliasOf ; NameOfCoef vr ; }
    }
    Constraint {
      { NameOfCoef V ; EntityType Region ; NameOfConstraint CurrentVelocity ; }
    }
  }

}

//-------------------------------------------------------------------------------------------

Formulation {

  { Name MagStaDyn_a_2D ; Type FemEquation ;
    Quantity {
      { Name a  ; Type Local  ; NameOfSpace Hcurl_a_2D ; }
      { Name ur ; Type Local  ; NameOfSpace Hregion_u_Mag_2D ; }
      { Name I  ; Type Global ; NameOfSpace Hregion_u_Mag_2D [I] ; }
      { Name U  ; Type Global ; NameOfSpace Hregion_u_Mag_2D [U] ; }

      { Name ir ; Type Local  ; NameOfSpace Hregion_i_Mag_2D ; }
      { Name Ub ; Type Global ; NameOfSpace Hregion_i_Mag_2D [Ub] ; }
      { Name Ib ; Type Global ; NameOfSpace Hregion_i_Mag_2D [Ib] ; }

      { Name Uz ; Type Global ; NameOfSpace Hregion_Z [Uz] ; }
      { Name Iz ; Type Global ; NameOfSpace Hregion_Z [Iz] ; }
    }
    Equation {
      Galerkin { [ nu[{d a}] * Dof{d a}  , {d a} ] ;
        In Domain ; Jacobian Vol ; Integration I1 ; }
      Galerkin { JacNL [ dhdb_NL[{d a}] * Dof{d a} , {d a} ] ;
        In DomainNL ; Jacobian Vol ; Integration I1 ; }

      // DO NOT REMOVE!!!
      // Keeping track of Dofs in auxiliar line of MB if Symmetry==1
      Galerkin {  [  0*Dof{d a} , {d a} ]  ;
        In Rotor_Bnd_MBaux; Jacobian Sur; Integration I1; }

      Galerkin { [ -nu[] * br[] , {d a} ] ;
        In DomainM ; Jacobian Vol ; Integration I1 ; }

      Galerkin { DtDof[ sigma[] * Dof{a} , {a} ] ;
        In DomainC ; Jacobian Vol ; Integration I1 ; }
      Galerkin { [ sigma[] * Dof{ur}, {a} ] ;
        In DomainC ; Jacobian Vol ; Integration I1 ; }

      Galerkin { [ -sigma[] * (Velocity[] *^ Dof{d a}) , {a} ] ;
        In DomainV ; Jacobian Vol ; Integration I1 ; }

      Galerkin { [ -js[] , {a} ] ;
        In DomainS ; Jacobian Vol ; Integration I1 ; }

      Galerkin { DtDof[ sigma[] * Dof{a} , {ur} ] ;
        In DomainC ; Jacobian Vol ; Integration I1 ; }
      Galerkin { [ sigma[] * Dof{ur} , {ur} ] ;
        In DomainC ; Jacobian Vol ; Integration I1 ; }
      GlobalTerm { [ Dof{I} , {U} ] ; In DomainC ; }

      Galerkin { [ -NbWires[]/SurfCoil[] * Dof{ir} , {a} ] ;
        In DomainB ; Jacobian Vol ; Integration I1 ; }
      Galerkin { DtDof [ AxialLength * NbWires[]/SurfCoil[] * Dof{a} , {ir} ] ;
        In DomainB ; Jacobian Vol ; Integration I1 ; }
      GlobalTerm { [ Dof{Ub}/SymmetryFactor , {Ib} ] ; In DomainB ; }
      Galerkin { [ Rb[]/SurfCoil[]* Dof{ir} , {ir} ] ;
        In DomainB ; Jacobian Vol ; Integration I1 ; } // Resistance term

      // GlobalTerm { [ Resistance[]  * Dof{Ib} , {Ib} ] ; In DomainB ; }
      // The above term can replace the resistance term:
      // if we have an estimation of the resistance of DomainB, via e.g. measurements
      // which is better to account for the end windings...

      If(Flag_Cir)
	GlobalTerm { NeverDt[ Dof{Uz}                , {Iz} ] ; In Resistance_Cir ; }
        GlobalTerm { NeverDt[ Resistance[] * Dof{Iz} , {Iz} ] ; In Resistance_Cir ; }

	GlobalTerm { [ Dof{Uz}                      , {Iz} ] ; In Inductance_Cir ; }
	GlobalTerm { DtDof [ Inductance[] * Dof{Iz} , {Iz} ] ; In Inductance_Cir ; }

	GlobalTerm { NeverDt[ Dof{Iz}        , {Iz} ] ; In Capacitance_Cir ; }
	GlobalTerm { DtDof [ Capacitance[] * Dof{Uz} , {Iz} ] ; In Capacitance_Cir ; }

	GlobalTerm { [ 0. * Dof{Iz} , {Iz} ] ; In DomainZt_Cir ; }
        GlobalTerm { [ 0. * Dof{Uz} , {Iz} ] ; In DomainZt_Cir ; }

        GlobalEquation {
          Type Network ; NameOfConstraint ElectricalCircuit ;
          { Node {I};  Loop {U};  Equation {I};  In DomainC ; }
          { Node {Ib}; Loop {Ub}; Equation {Ub}; In DomainB ; }
          { Node {Iz}; Loop {Uz}; Equation {Uz}; In DomainZt_Cir ; }
         }
      EndIf
    }
  }

 //-------------------------------------------------------------------------------------------
// Mechanics

  { Name Mechanical ; Type FemEquation ;
    Quantity {
      { Name V ; Type Global ; NameOfSpace Velocity [V] ; } // velocity
      { Name P ; Type Global ; NameOfSpace Position [P] ; } // position
    }
    Equation {
      GlobalTerm { DtDof [ Inertia * Dof{V} , {V} ] ; In DomainKin ; }
      GlobalTerm { [ Friction[] * Dof{V} , {V} ] ; In DomainKin ; }
      GlobalTerm { [        Torque_mec[] , {V} ] ; In DomainKin ; }
      GlobalTerm { [       -Torque_mag[] , {V} ] ; In DomainKin ; }

      GlobalTerm { DtDof [ Dof{P} , {P} ] ; In DomainKin ; }
      GlobalTerm {       [-Dof{V} , {P} ] ; In DomainKin ; }
    }
  }

}

//-----------------------------------------------------------------------------------------

Resolution {

  { Name Analysis ;
    System {
      If(Flag_AnalysisType==2)
        { Name A ; NameOfFormulation MagStaDyn_a_2D ; Type ComplexValue ; Frequency Freq ; }
      EndIf
      If(Flag_AnalysisType<2)
        { Name A ; NameOfFormulation MagStaDyn_a_2D ; }
        If(!Flag_ImposedSpeed) // Full dynamics
          { Name M ; NameOfFormulation Mechanical ; }
        EndIf
      EndIf
    }
    Operation {
      CreateDir[ResDir];
      If(Clean_Results==1 && variableFrequencyLoop == initialFrequencyLoop)
	// FIXME == > variable controlling loop in Onelab

	DeleteFile[StrCat[ResDir,"temp",ExtGnuplot]];
	DeleteFile[StrCat[ResDir,"Tr",ExtGnuplot]];
	DeleteFile[StrCat[ResDir,"Ts",ExtGnuplot]];
	DeleteFile[StrCat[ResDir,"Tmb",ExtGnuplot]];
	DeleteFile[StrCat[ResDir,"Ua",ExtGnuplot]];
	DeleteFile[StrCat[ResDir,"Ub",ExtGnuplot]];
	DeleteFile[StrCat[ResDir,"Uc",ExtGnuplot]];
	DeleteFile[StrCat[ResDir,"Ia",ExtGnuplot]];
	DeleteFile[StrCat[ResDir,"Ib",ExtGnuplot]];
	DeleteFile[StrCat[ResDir,"Ic",ExtGnuplot]];
	DeleteFile[StrCat[ResDir,"Flux_a",ExtGnuplot]];
	DeleteFile[StrCat[ResDir,"Flux_b",ExtGnuplot]];
	DeleteFile[StrCat[ResDir,"Flux_c",ExtGnuplot]];
	DeleteFile[StrCat[ResDir,"Flux_d",ExtGnuplot]];
	DeleteFile[StrCat[ResDir,"Flux_q",ExtGnuplot]];
	DeleteFile[StrCat[ResDir,"Flux_0",ExtGnuplot]];
	DeleteFile[StrCat[ResDir,"JL",ExtGnuplot]];
	DeleteFile[StrCat[ResDir,"JL_Fe",ExtGnuplot]];
	DeleteFile[StrCat[ResDir,"P",ExtGnuplot]];
	DeleteFile[StrCat[ResDir,"V",ExtGnuplot]];
	DeleteFile[StrCat[ResDir,"Irotor",ExtGnuplot]];
    Printf('blubb11111');
      EndIf

      If(Flag_AnalysisType==0 || Flag_AnalysisType==2)
        If(Flag_AnalysisType==2)
          SetTime[variableFrequencyLoop];
        EndIf

        InitMovingBand2D[MB] ;
        MeshMovingBand2D[MB] ;
        Printf('before2');
        InitSolution[A] ;
        Printf('after2');
        If(Flag_ParkTransformation && Flag_SrcType_Stator==1)
          PostOperation[ThetaPark_IABC] ;
        EndIf
        If(!Flag_NL)
          Generate[A] ; Solve[A] ;
        EndIf
        If(Flag_NL)
          IterativeLoop[Nb_max_iter, stop_criterion, relaxation_factor]{
            GenerateJac[A] ; SolveJac[A] ; }
        EndIf
        Printf('point 1');
        SaveSolution[A] ;
        Printf('Point2');
        If(Flag_PrintFields)
            Printf('get_localfiels');
          PostOperation[Get_LocalFields] ;
        EndIf
        Printf('get_globalfiels');
        PostOperation[Get_GlobalQuantities] ;
        If(Flag_AnalysisType==0)
          PostOperation[Get_Torque];
        EndIf
        If(Flag_AnalysisType==2)
          PostOperation[Get_Torque_cplx];
        EndIf
      EndIf // End Flag_AnalysisType==0 (Static) Flag_AnalysisType==2 (Frequency)

      If(Flag_AnalysisType==1)
        InitSolution[A];
        SaveSolution[A];
        InitMovingBand2D[MB];
		

        If(!Flag_ImposedSpeed) // Full dynamics
          InitSolution[M];
          InitSolution[M]; // Twice for avoiding warning (a = d_t^2 x)
          PostOperation[Mechanical] ;
        EndIf
        
        // take care of non-zero initial time
        Test[ $Time > time0 ]{
          If(!Flag_ImposedSpeed) // not tested yet
            Evaluate[ $PreviousPosition = 0 ];
            ChangeOfCoordinates[ NodesOf[Rotor_Moving], RotatePZ[delta_theta[]]];
          Else // this has been tested
            ChangeOfCoordinates[ NodesOf[Rotor_Moving], RotatePZ[$Time*wr]];
          EndIf
        }
        MeshMovingBand2D[MB];

        TimeLoopTheta[time0, timemax, dtime, 1.]{
	  // Euler implicit (1) -- Crank-Nicolson (0.5)
	  // FIXME like this theta cannot be controlled by the user
          If(Flag_ParkTransformation && Flag_SrcType_Stator==1)
            PostOperation[ThetaPark_IABC];
          EndIf
          If(!Flag_NL)
            Generate[A]; Solve[A];
          EndIf
          If(Flag_NL)
            IterativeLoop[Nb_max_iter, stop_criterion, relaxation_factor] {
              GenerateJac[A] ; SolveJac[A] ; }
          EndIf
		  SaveSolution[A];

          If(Flag_PrintFields)
                      Printf('get_localfiels');
            PostOperation[Get_LocalFields] ;			
          EndIf
          Test[ $Time > time0 ]{
                      Printf('get_globalfiels');
            PostOperation[Get_GlobalQuantities];
            PostOperation[Get_Torque] ;
          }

          {
            Evaluate[ $Tstator = 0 ];
            Evaluate[ $Trotor = 0 ];
          }
          If(!Flag_ImposedSpeed)
            Generate[M]; Solve[M]; SaveSolution[M];
            PostOperation[Mechanical] ;
          EndIf
		  Print[{delta_theta[]}, Format "delta_theta[] = %g is greater than 1"];
		  //Print[{js[]}, Format "delta_theta[] = %g is greater than 1"];
          ChangeOfCoordinates[ NodesOf[Rotor_Moving], RotatePZ[delta_theta[]]];
          If(!Flag_ImposedSpeed)
            // Keep track of previous position
            Evaluate[ $PreviousPosition = $Position ];
          EndIf
          MeshMovingBand2D[MB] ;
        }
        Printf('before');
        SaveSolution[A]; //save soln in the end of the interval only
        Printf('after');
      EndIf // End Flag_AnalysisType==1 (Time domain)
    }
  }
}

//-------------------------------------------------------------------------------------------

PostProcessing {

  { Name MagStaDyn_a_2D ; NameOfFormulation MagStaDyn_a_2D ;
   PostQuantity {
     { Name a  ; Value { Term { [ {a} ] ; In Domain ; Jacobian Vol ; } } }
     { Name az ; Value { Term { [ CompZ[{a}] ] ; In Domain ; Jacobian Vol ; } } }

     { Name b  ; Value { Term { [ {d a} ] ; In Domain ; Jacobian Vol ; } } }
     { Name boundary  ; Value { Term { [ 1 ] ; In DomainPlotMovingGeo ; Jacobian Vol ; } } } // Dummy value - for visualization
     { Name b_radial ;
       Value { Term { [ {d a}* Vector[  Cos[AngularPosition[]#4], Sin[#4], 0.] ] ;
	   In Domain ; Jacobian Vol ; } } }
     { Name b_tangent ;
       Value { Term { [ {d a}* Vector[ -Sin[AngularPosition[]#4], Cos[#4], 0.] ] ;
	   In Domain ; Jacobian Vol ; } } }

     { Name js ; Value { Term { [ js[] ] ; In DomainS ; Jacobian Vol ; } } }

     { Name js_A ; Value { Term { [ js[] ] ; In PhaseA ; Jacobian Vol ; } } }
     { Name js_B ; Value { Term { [ js[] ] ; In PhaseB ; Jacobian Vol ; } } }
     { Name js_C ; Value { Term { [ js[] ] ; In PhaseC ; Jacobian Vol ; } } }

     { Name br ; Value { Term { [ br[] ] ; In DomainM ; Jacobian Vol ; } } }

     { Name j  ; Value {
         Term { [ -sigma[]*(Dt[{a}]+{ur}) ]        ; In DomainC ; Jacobian Vol ; }
         Term { [  sigma[]*(Velocity[] *^ {d a}) ] ; In DomainV ; Jacobian Vol ; }
       }
     }
     { Name ir ; Value { Term { [ {ir} ] ; In Inds ; Jacobian Vol ; } } }

     { Name jz ; Value {
         Term { [ CompZ[-sigma[]*(Dt[{a}]+{ur})] ]       ; In DomainC ; Jacobian Vol ; }
         Term { [ CompZ[ sigma[]*(Velocity[]*^{d a}) ] ] ; In DomainV ; Jacobian Vol ; }
       }
     }

     { Name rhoj2 ;
       Value {
         Term { [ sigma[]*SquNorm[ Dt[{a}]+{ur}] ] ;
	   In Region[{DomainC,-DomainV}] ; Jacobian Vol ; }
         Term { [ sigma[]*SquNorm[ Dt[{a}]+{ur}-Velocity[]*^{d a} ] ] ;
	   In DomainV ; Jacobian Vol ; }
         Term { [ 1./sigma[]*SquNorm[ IA[]*{ir} ] ] ; In PhaseA  ; Jacobian Vol ; }
         Term { [ 1./sigma[]*SquNorm[ IB[]*{ir} ] ] ; In PhaseB  ; Jacobian Vol ; }
         Term { [ 1./sigma[]*SquNorm[ IC[]*{ir} ] ] ; In PhaseC  ; Jacobian Vol ; }
       }
     }

     { Name JouleLosses ;
       Value {
         Integral { [ sigma[] * SquNorm[ Dt[{a}]+{ur} ] ] ;
	   In Region[{DomainC,-DomainV}] ; Jacobian Vol ; Integration I1 ; }
         Integral { [ sigma[] * SquNorm[ Dt[{a}]+{ur}-Velocity[]*^{d a} ] ] ;
	   In DomainV ; Jacobian Vol ; Integration I1 ; }
         Integral { [ 1./sigma[]*SquNorm[ IA[]*{ir} ] ] ;
	   In PhaseA ; Jacobian Vol ; Integration I1 ; }
         Integral { [ 1./sigma[]*SquNorm[ IB[]*{ir} ] ] ;
	   In PhaseB  ; Jacobian Vol ; Integration I1 ; }
         Integral { [ 1./sigma[]*SquNorm[ IC[]*{ir} ] ] ;
	   In PhaseC  ; Jacobian Vol ; Integration I1 ; }
       }
     }

     { Name Flux ;
       Value {
	 Integral { [ SymmetryFactor*AxialLength*Idir[]*NbWires[]/SurfCoil[]* CompZ[{a}] ] ;
           In Inds  ; Jacobian Vol ; Integration I1 ; } } }
     { Name Force_vw ;
       // Force computation by Virtual Works
       Value {
         Integral {
           Type Global ; [ 0.5 * nu[] * VirtualWork [{d a}] * AxialLength ];
           In ElementsOf[Rotor_Airgap, OnOneSideOf Rotor_Bnd_MB];
	   Jacobian Vol ; Integration I1 ; }
       }
     }

     { Name Torque_vw ; Value {
	 // Torque computation via Virtual Works
         Integral { Type Global ;
           [ CompZ[ 0.5 * nu[] * XYZ[] /\ VirtualWork[{d a}] ] * AxialLength ];
           In ElementsOf[Rotor_Airgap, OnOneSideOf Rotor_Bnd_MB];
	   Jacobian Vol ; Integration I1 ; }
       }
     }

     { Name Torque_Maxwell ;
       // Torque computation via Maxwell stress tensor
       Value {
         Integral {
           // \int_S (\vec{r} \times (T_max \vec{n}) ) / ep
           // with ep = |S| / (2\pi r_avg)
           [ CompZ [ XYZ[] /\ (T_max[{d a}] * XYZ[]) ] * 2*Pi*AxialLength/SurfaceArea[] ] ;
           In Domain ; Jacobian Vol  ; Integration I1; }
       }
     }

     { Name Torque_Maxwell_cplx ;
       // Torque computation via Maxwell stress tensor - frequency domain
       Value {
         Integral {
           [ CompZ [ XYZ[] /\ (T_max_cplx[{d a}] * XYZ[]) ] * 2*Pi*AxialLength/SurfaceArea[] ] ;
           In Domain ; Jacobian Vol  ; Integration I1; }
       }
     }

     { Name Torque_Maxwell_cplx_2f ;
       // Torque computation via Maxwell stress tensor, component at twice the frequency
       Value {
         Integral {
           [ CompZ [ XYZ[] /\ (T_max_cplx_2f[{d a}] * XYZ[]) ] * 2*Pi*AxialLength/SurfaceArea[] ] ;
           In Domain ; Jacobian Vol  ; Integration I1; }
       }
     }

     { Name ComplexPower ;
       // S = P + i*Q
       Value {
         Integral { [ Complex[ sigma[]*SquNorm[Dt[{a}]+{ur}], nu[]*SquNorm[{d a}] ] ] ;
           In Region[{DomainC,-DomainV}] ; Jacobian Vol ; Integration I1 ; }
         Integral { [ Complex[ sigma[]*SquNorm[Dt[{a}]+{ur}-Velocity[]*^{d a}], nu[]*SquNorm[{d a}] ] ] ;
           In DomainV ; Jacobian Vol ; Integration I1 ; }
       }
     }

     { Name U ; Value {
         Term { [ {U} ]   ; In DomainC ; }
         Term { [ {Ub} ]  ; In DomainB ; }
         Term { [ {Uz} ]  ; In DomainZt_Cir ; }
     } }

     { Name I ; Value {
         Term { [ {I} ]   ; In DomainC ; }
         Term { [ {Ib} ]  ; In DomainB ; }
         Term { [ {Iz} ]  ; In DomainZt_Cir ; }
     } }

     { Name S ; Value {
         Term { [ {U}*Conj[{I}] ]    ; In DomainC ; }
         Term { [ {Ub}*Conj[{Ib}] ]  ; In DomainB ; }
         Term { [ {Uz}*Conj[{Iz}] ]  ; In DomainZt_Cir ; }
     } }

     { Name Velocity  ; Value {
         Term { [ Velocity[] ] ; In Domain ; Jacobian Vol ; }
       }
     }

     { Name RotorPosition_deg ;
       Value { Term { Type Global; [ RotorPosition_deg[] ] ; In DomainDummy ; } } }
     { Name Theta_Park_deg ;
       Value { Term { Type Global; [ Theta_Park_deg[] ] ; In DomainDummy ; } } }
     { Name IA  ;
       Value { Term { Type Global; [ IA[] ] ; In DomainDummy ; } } }
     { Name IB  ;
       Value { Term { Type Global; [ IB[] ] ; In DomainDummy ; } } }
     { Name IC  ;
       Value { Term { Type Global; [ IC[] ] ; In DomainDummy ; } } }
     { Name Flux_d  ;
       Value { Term { Type Global; [ CompX[Flux_dq0[]] ] ; In DomainDummy ; } } }
     { Name Flux_q  ;
       Value { Term { Type Global; [ CompY[Flux_dq0[]] ] ; In DomainDummy ; } } }
     { Name Flux_0  ;
       Value { Term { Type Global; [ CompZ[Flux_dq0[]] ] ; In DomainDummy ; } } }
   }
 }

 { Name Mechanical ; NameOfFormulation Mechanical ;
   PostQuantity {
     { Name P ; Value { Term { [ {P} ]  ; In DomainKin ; } } } // Position [rad]
     { Name Pdeg ; Value { Term { [ {P}*180/Pi ]  ; In DomainKin ; } } } // Position [deg]
     { Name V ; Value { Term { [ {V} ]  ; In DomainKin ; } } } // Velocity [rad/s]
     { Name Vrpm ; Value { Term { [ {V}*30/Pi ]  ; In DomainKin ; } } } // Velocity [rpm]
   }
 }

}

//-------------------------------------------------------------------------------------------

po      = StrCat["Output - Electromagnetics/", ResId];
poI     = StrCat[po,"0Current [A]/"];
poV     = StrCat[po,"1Voltage [V]/"];
//Printf("mutta[] = %g ", poV);
poF     = StrCat[po,"2Flux linkage [Vs]/"];
poJL    = StrCat[po,"3Joule Losses [W]/"];
po_mec  = StrCat["Output - Mechanics/", ResId];
po_mecT = StrCat[po_mec,"0Torque [Nm]/"];

//-------------------------------------------------------------------------------------------

PostOperation Get_Mech UsingPost Mechanical {
  Print[ P, OnRegion DomainKin, Format Table,
	 File > StrCat[ResDir,"P", ExtGnuplot], LastTimeStepOnly] ;
	// SendToServer StrCat[po_mec,"11Position [rad]"]{0}, Color "Ivory"] ;
}

PostOperation Get_LocalFields UsingPost MagStaDyn_a_2D {
  Print[ ir, OnElementsOf Stator_Inds,
	 File StrCat[ResDir,"ir_stator",ExtGmsh], LastTimeStepOnly,
	 AppendTimeStepToFileName Flag_SaveAllSteps] ;
  Print[ ir, OnElementsOf Rotor_Inds,
	 File StrCat[ResDir,"ir_rotor",ExtGmsh], LastTimeStepOnly,
	 AppendTimeStepToFileName Flag_SaveAllSteps] ;
  Print[ jz, OnElementsOf DomainC,
	 File StrCat[ResDir,"jz",ExtGmsh], LastTimeStepOnly,
	 AppendTimeStepToFileName Flag_SaveAllSteps ] ;
  Print[ js, OnElementsOf DomainS,
	 File StrCat[ResDir,"js",ExtGmsh], LastTimeStepOnly,
	 AppendTimeStepToFileName Flag_SaveAllSteps ] ;

    Printf('hallo');
  Print[ js_A, OnPoint { 0.056737197, 0.015202686, 0.0}, Format TimeTable, File > StrCat[ResDir,"js_A",ExtGnuplot], LastTimeStepOnly] ;
  Print[ js_B, OnPoint { 0.048115900, 0.033691116, 0.0}, Format TimeTable, File > StrCat[ResDir,"js_B",ExtGnuplot], LastTimeStepOnly] ;
  Print[ js_C, OnPoint { 0.015202686, 0.056737197, 0.0}, Format TimeTable, File > StrCat[ResDir,"js_C",ExtGnuplot], LastTimeStepOnly] ;

  Print[ b,  OnElementsOf Domain,
	 File StrCat[ResDir,"b",ExtGmsh], LastTimeStepOnly,
	 AppendTimeStepToFileName Flag_SaveAllSteps] ;
  Print[ boundary, OnElementsOf DomainPlotMovingGeo,
	 File StrCat[ResDir,"bnd",ExtGmsh], LastTimeStepOnly,
	 AppendTimeStepToFileName Flag_SaveAllSteps] ;
  Echo[ Str["For i In {PostProcessing.NbViews-1:0:-1}",
            "  If(!StrCmp(View[i].Name, 'boundary'))",
            "    View[i].ShowElement=1;",
            "   EndIf",
            "EndFor"], File "res/tmp.geo", LastTimeStepOnly] ;
  Print[ az, OnElementsOf Domain,
	 File StrCat[ResDir,"a",ExtGmsh], LastTimeStepOnly,
	 AppendTimeStepToFileName Flag_SaveAllSteps ] ;
}

PostOperation Get_GlobalQuantities UsingPost MagStaDyn_a_2D {
            Printf('step1');
  If(!Flag_Cir)
              Printf('step2');
    If(!Flag_ParkTransformation)
      Print[ I, OnRegion PhaseA_pos, Format Table,
	     File > StrCat[ResDir,"Ia",ExtGnuplot], LastTimeStepOnly,
	     SendToServer StrCat[poI,"A"]{0}, Color "Pink" ];
      If(NbrPhases==3)
        Print[ I, OnRegion PhaseB_pos, Format Table,
	       File > StrCat[ResDir,"Ib",ExtGnuplot], LastTimeStepOnly,
	       SendToServer StrCat[poI,"B"]{0}, Color "Yellow" ];
        Print[ I, OnRegion PhaseC_pos, Format Table,
	       File > StrCat[ResDir,"Ic",ExtGnuplot], LastTimeStepOnly,
	       SendToServer StrCat[poI,"C"]{0}, Color "LightGreen" ];
      EndIf
    EndIf
    Print[{U}, Format "gagaga[] = %g is greater than 1"];
    Print[ U, OnRegion PhaseA_pos, Format Table,
	   File > StrCat[ResDir,"Ua",ExtGnuplot], LastTimeStepOnly,
	   SendToServer StrCat[poV,"A"]{0}, Color "Pink" ];
    If(NbrPhases==3)
      Print[ U, OnRegion PhaseB_pos, Format Table,
	     File > StrCat[ResDir,"Ub",ExtGnuplot], LastTimeStepOnly,
	     SendToServer StrCat[poV,"B"]{0}, Color "Yellow" ];
      Print[ U, OnRegion PhaseC_pos, Format Table,
	     File > StrCat[ResDir,"Uc",ExtGnuplot], LastTimeStepOnly,
	     SendToServer StrCat[poV,"C"]{0}, Color "LightGreen" ];
    EndIf
  EndIf
  If(Flag_Cir && Flag_SrcType_StatorA==2)
  Printf('step3');
  Print[{Flag_SrcType_Stator}, Format "gagaga[] = %f is greater than 1"];
     // Print[ I, OnRegion Input1, Format Table];
    Print[ I, OnRegion Input1, Format Table,
	   File > StrCat[ResDir,"Ia",ExtGnuplot], LastTimeStepOnly,
	   SendToServer StrCat[poI,"A"]{0}, Color "Pink" ];
    Print[ U, OnRegion Input1, Format Table,
	   File > StrCat[ResDir,"Ua",ExtGnuplot], LastTimeStepOnly,
	   SendToServer StrCat[poV,"A"]{0}, Color "Pink" ];
  EndIf
  If(Flag_Cir && Flag_SrcType_StatorB==2)
  Printf('step4');
  Print[{Flag_SrcType_Stator}, Format "gagaga2[] = %g is greater than 1"];
    Print[ I, OnRegion Input2, Format Table,
	   File > StrCat[ResDir,"Ib",ExtGnuplot], LastTimeStepOnly,
	   SendToServer StrCat[poI,"B"]{0}, Color "Yellow" ];
    Print[ U, OnRegion Input2, Format Table,
	   File > StrCat[ResDir,"Ub",ExtGnuplot], LastTimeStepOnly,
	   SendToServer StrCat[poV,"B"]{0}, Color "Yellow" ];
  EndIf
  If(Flag_Cir && Flag_SrcType_StatorB==2)
  Printf('step5');
  Print[{delta_theta[]}, Format "gagaga3[] = %g is greater than 1"];
    Print[ I, OnRegion Input3, Format Table,
	   File > StrCat[ResDir,"Ic",ExtGnuplot], LastTimeStepOnly,
	   SendToServer StrCat[poI,"C"]{0}, Color "LightGreen" ];
    Print[ U, OnRegion Input3, Format Table,
	   File > StrCat[ResDir,"Uc",ExtGnuplot], LastTimeStepOnly,
	   SendToServer StrCat[poV,"C"]{0}, Color "LightGreen" ];
  EndIf
  If(Flag_Cir && Flag_SrcType_StatorA==0)
  Printf('step6');
    Print[ I, OnRegion R1, Format Table,
	   File > StrCat[ResDir,"Ia",ExtGnuplot], LastTimeStepOnly,
	   SendToServer StrCat[poI,"A"]{0}, Color "Pink" ];
    Print[ U, OnRegion R1, Format Table,
	   File > StrCat[ResDir,"Ua",ExtGnuplot], LastTimeStepOnly,
	   SendToServer StrCat[poV,"A"]{0}, Color "Pink" ];
  EndIf
  If(Flag_Cir && Flag_SrcType_StatorB==0)
  Printf('step7');
    Print[ I, OnRegion R2, Format Table,
	   File > StrCat[ResDir,"Ib",ExtGnuplot], LastTimeStepOnly,
	   SendToServer StrCat[poI,"B"]{0}, Color "Yellow" ];
    Print[ U, OnRegion R2, Format Table,
	   File > StrCat[ResDir,"Ub",ExtGnuplot], LastTimeStepOnly,
	   SendToServer StrCat[poV,"B"]{0}, Color "Yellow" ];
  EndIf
  If(Flag_Cir && Flag_SrcType_StatorC==0)
    Print[ I, OnRegion R3, Format Table,
	   File > StrCat[ResDir,"Ic",ExtGnuplot], LastTimeStepOnly,
	   SendToServer StrCat[poI,"C"]{0}, Color "LightGreen" ];
    Print[ U, OnRegion R3, Format Table,
	   File > StrCat[ResDir,"Uc",ExtGnuplot], LastTimeStepOnly,
	   SendToServer StrCat[poV,"C"]{0}, Color "LightGreen" ];
  EndIf
Printf('step8');
  Print[ I, OnRegion RotorC, Format Table,
	 File > StrCat[ResDir,"Irotor",ExtGnuplot], LastTimeStepOnly,
	 SendToServer StrCat[poI,"rotor"]{0}, Color "LightCyan" ];

  If(Flag_SrcType_Stator)
    Print[ Flux[PhaseA], OnGlobal, Format TimeTable,
	   File > StrCat[ResDir,"Flux_a",ExtGnuplot], LastTimeStepOnly,
           StoreInVariable $Flux_a, SendToServer StrCat[poF,"A"]{0},  Color "Pink" ];
    If(NbrPhases==3)
      Print[ Flux[PhaseB], OnGlobal, Format TimeTable,
	     File > StrCat[ResDir,"Flux_b",ExtGnuplot], LastTimeStepOnly,
             StoreInVariable $Flux_b, SendToServer StrCat[poF,"B"]{0},  Color "Yellow" ];
      Print[ Flux[PhaseC], OnGlobal, Format TimeTable,
	     File > StrCat[ResDir,"Flux_c",ExtGnuplot], LastTimeStepOnly,
             StoreInVariable $Flux_c, SendToServer StrCat[poF,"C"]{0}, Color "LightGreen"];
    EndIf
    If(Flag_ParkTransformation && Flag_SrcType_Stator)
      Print[ Flux_d, OnRegion DomainDummy, Format TimeTable,
	     File > StrCat[ResDir,"Flux_d",ExtGnuplot], LastTimeStepOnly,
	     SendToServer StrCat[poF,"d"]{0}, Color "LightYellow" ];
      Print[ Flux_q, OnRegion DomainDummy, Format TimeTable,
	     File > StrCat[ResDir,"Flux_q",ExtGnuplot], LastTimeStepOnly,
	     SendToServer StrCat[poF,"q"]{0}, Color "LightYellow" ];
      Print[ Flux_0, OnRegion DomainDummy, Format TimeTable,
	     File > StrCat[ResDir,"Flux_0",ExtGnuplot], LastTimeStepOnly,
	     SendToServer StrCat[poF,"0"]{0}, Color "LightYellow" ];
    EndIf
  EndIf

  Print[ JouleLosses[Rotor], OnGlobal, Format TimeTable,
	 File > StrCat[ResDir,"JL",ExtGnuplot], LastTimeStepOnly,
	 SendToServer StrCat[poJL,"rotor"]{0}, Color "LightYellow" ];
  Print[ JouleLosses[Rotor_Fe], OnGlobal, Format TimeTable,
	 File > StrCat[ResDir,"JL_Fe",ExtGnuplot], LastTimeStepOnly,
	 SendToServer StrCat[poJL,"rotor_fe"]{0}, Color "LightYellow" ];
}

PostOperation Get_Torque UsingPost MagStaDyn_a_2D {
  Print[ Torque_Maxwell[Rotor_Airgap], OnGlobal, Format TimeTable,
    File > StrCat[ResDir,"Tr",ExtGnuplot], LastTimeStepOnly, StoreInVariable $Trotor,
	 SendToServer StrCat[po_mecT, "rotor"]{0}, Color "Ivory" ];
  Print[ Torque_Maxwell[Stator_Airgap], OnGlobal, Format TimeTable,
    File > StrCat[ResDir,"Ts",ExtGnuplot], LastTimeStepOnly, StoreInVariable $Tstator,
	 SendToServer StrCat[po_mecT, "stator"]{0}, Color "Ivory" ];
}

PostOperation Get_Torque_cplx UsingPost MagStaDyn_a_2D {
  Print[ Torque_Maxwell_cplx[Rotor_Airgap], OnGlobal, Format TimeTable,
	 File > StrCat[ResDir,"Tr",ExtGnuplot], StoreInVariable $Trotor,
	 SendToServer StrCat[po_mecT, "rotor"]{0}, Color "Ivory" ];
  Print[ Torque_Maxwell_cplx[Stator_Airgap], OnGlobal, Format TimeTable,
	 File > StrCat[ResDir,"Ts",ExtGnuplot], StoreInVariable $Tstator,
	 SendToServer StrCat[po_mecT,"stator"]{0}, Color "Ivory" ];
}

PostOperation Mechanical UsingPost Mechanical {
  Print[ P, OnRegion DomainKin, Format Table,
	 File > StrCat[ResDir,"P", ExtGnuplot], LastTimeStepOnly, StoreInVariable $Position,
	 SendToServer StrCat[po_mec,"11Position [rad]"]{0}, Color "Ivory"] ;
  Print[ Pdeg, OnRegion DomainKin, Format Table,
	 File > StrCat[ResDir,"P_deg", ExtGnuplot], LastTimeStepOnly,
	 SendToServer StrCat[po_mec,"10Position [deg]"]{0}, Color "Ivory"] ;
  Print[ V, OnRegion DomainKin, Format Table,
	 File > StrCat[ResDir,"V", ExtGnuplot], LastTimeStepOnly,
	 SendToServer StrCat[po_mec,"21Velocity [rad\s]"]{0}, Color "Ivory"] ;//MediumPurple1
  Print[ Vrpm, OnRegion DomainKin, Format Table,
	 File > StrCat[ResDir,"Vrpm", ExtGnuplot], LastTimeStepOnly,
	 SendToServer StrCat[po_mec,"20Velocity [rpm]"]{0}, Color "Ivory"] ;//MediumPurple1
}

If (Flag_ParkTransformation)
PostOperation ThetaPark_IABC UsingPost MagStaDyn_a_2D {
  Print[ RotorPosition_deg, OnRegion DomainDummy, Format Table, LastTimeStepOnly,
	 File StrCat[ResDir, "temp",ExtGnuplot],
	 SendToServer StrCat[po,"10Rotor position"]{0}, Color "LightYellow" ];
  Print[ Theta_Park_deg, OnRegion DomainDummy, Format Table, LastTimeStepOnly,
	 File StrCat[ResDir, "temp",ExtGnuplot],
	 SendToServer StrCat[po,"11Theta park"]{0}, Color "LightYellow" ];
  Print[ IA, OnRegion DomainDummy, Format Table, LastTimeStepOnly,
	 File StrCat[ResDir,"temp",ExtGnuplot],
	 SendToServer StrCat[poI,"A"]{0}, Color "Pink" ];
  Print[ IB, OnRegion DomainDummy, Format Table, LastTimeStepOnly,
	 File StrCat[ResDir,"temp",ExtGnuplot],
	 SendToServer StrCat[poI,"B"]{0}, Color "Yellow" ];
  Print[ IC, OnRegion DomainDummy, Format Table, LastTimeStepOnly,
	 File StrCat[ResDir,"temp",ExtGnuplot],
	 SendToServer StrCat[poI,"C"]{0}, Color "LightGreen"  ];
}
EndIf
