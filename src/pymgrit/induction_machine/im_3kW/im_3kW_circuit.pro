// Authors - J. Gyselinck, R.V. Sabariego (2013)
//
// Circuit for induction motor
//

Group{
  // Dummy numbers for circuit definition

  Input1 = Region[10001] ;
  Input2 = Region[10002] ;
  Input3 = Region[10003] ;

  R1 = Region[55551] ;
  R2 = Region[55552] ;
  R3 = Region[55553] ;

  L1 = Region[55561] ;
  L2 = Region[55562] ;
  L3 = Region[55563] ;

  For k In {1:nbRotorBars}
    Rers~{k} = Region[{(60000+k)}]; // resistance per endring segment
    All_EndRingResistancesRotor += Region[ Rers~{k} ] ;
    Lers~{k} = Region[{(70000+k)}]; // inductance per end ring segment
    All_EndRingInductancesRotor += Region[ Lers~{k} ] ;
  EndFor

  Resistance_Cir  = Region[{ }];
  Inductance_Cir  = Region[{L1, L2, L3}];

  If(Flag_Cir_RotorCage)
    Resistance_Cir  += Region[{ All_EndRingResistancesRotor }];
    Inductance_Cir  += Region[{ All_EndRingInductancesRotor }];
  EndIf

  DomainZ_Cir = Region[ {Resistance_Cir, Inductance_Cir} ];
  DomainSource_Cir = Region[ {Input1, Input2, Input3} ] ;
  DomainZt_Cir    = Region[ {DomainZ_Cir, DomainSource_Cir} ];
}

// --------------------------------------------------------------------------
// --------------------------------------------------------------------------

Function {
  For k In {1:nbRotorBars}
    NB1~{k} =  400+k; // first node number for each rotor bar
    NB2~{k} =  500+k; // second node number for each rotor bar
  EndFor
  For k In {1:nbRotorBars}
    NR1~{k} = NB1~{k}; // first node number for each endring resistance
    k2 = (k<nbRotorBars) ? k+1 : 1.;
    NR2~{k} = (k<nbRotorBars && SymmetryFactor<4) ? NB1~{k2} : NB2~{1} ; // second node number for each endring resistance
    NL1~{k} = NB2~{k}; // first node number for each endring inductance
    NL2~{k} = (k<nbRotorBars && SymmetryFactor<4) ? NB2~{k2} : NB1~{1} ; // second node number for each endring resistance
  EndFor

  Inductance[Region[{L1, L2, L3}]]  = Ls ; // endwinding reactance per phase
  Resistance[All_EndRingResistancesRotor] = R_endring_segment;
  Inductance[All_EndRingInductancesRotor] = L_endring_segment;
}


// --------------------------------------------------------------------------

Constraint {
  If (SymmetryFactor<4 && !Flag_Cir_RotorCage)
    { Name ElectricalCircuit ; Type Network ;
      Case Circuit1 {
        { Region Input1        ; Branch {100,101} ; }
        { Region L1            ; Branch {101,102} ; }
        { Region Stator_Ind_Ap ; Branch {102,103} ; }
        { Region Stator_Ind_Am ; Branch {100,103} ; }
      }
      Case Circuit2 {
        { Region Input2        ; Branch {200,201} ; }
        { Region L2            ; Branch {201,202} ; }
        { Region Stator_Ind_Bp ; Branch {202,203} ; }
        { Region Stator_Ind_Bm ; Branch {200,203} ; }
      }
      Case Circuit3 {
        { Region Input3        ; Branch {300,301} ; }
        { Region L3            ; Branch {301,302} ; }
        { Region Stator_Ind_Cp ; Branch {302,303} ; }
        { Region Stator_Ind_Cm ; Branch {300,303} ; }
      }
    }
  EndIf

  If(SymmetryFactor==4 && !Flag_Cir_RotorCage) // Only one physical region in geo allow per branch
    { Name ElectricalCircuit ; Type Network ;
      Case Circuit1 {
        { Region Input1        ; Branch {100,101} ; }
        { Region L1            ; Branch {101,102} ; }
        { Region Stator_Ind_Ap ; Branch {102,100} ; }
      }
      Case Circuit2 {
        { Region Input2        ; Branch {200,201} ; }
        { Region L2            ; Branch {201,202} ; }
        { Region Stator_Ind_Bp ; Branch {202,200} ; }
      }
      Case Circuit3 {
        { Region Input3        ; Branch {300,301} ; }
        { Region L3            ; Branch {301,302} ; }
        { Region Stator_Ind_Cm ; Branch {300,302} ; }
      }
    }
  EndIf

  If (SymmetryFactor<4 && Flag_Cir_RotorCage)
    { Name ElectricalCircuit ; Type Network ;
      Case Circuit1 {
        { Region Input1        ; Branch {100,101} ; }
        { Region L1            ; Branch {101,102} ; }
        { Region Stator_Ind_Ap ; Branch {102,103} ; }
        { Region Stator_Ind_Am ; Branch {100,103} ; }
      }
      Case Circuit2 {
        { Region Input2        ; Branch {200,201} ; }
        { Region L2            ; Branch {201,202} ; }
        { Region Stator_Ind_Bp ; Branch {202,203} ; }
        { Region Stator_Ind_Bm ; Branch {200,203} ; }
      }
      Case Circuit3 {
        { Region Input3        ; Branch {300,301} ; }
        { Region L3            ; Branch {301,302} ; }
        { Region Stator_Ind_Cp ; Branch {302,303} ; }
        { Region Stator_Ind_Cm ; Branch {300,303} ; }
      }
      Case Circuit4 {
        For k In {1:nbRotorBars}
          { Region Rotor_Bar~{k} ; Branch {NB1~{k}, NB2~{k}} ; }
          { Region Rers~{k} ;      Branch {NR1~{k}, NR2~{k}} ; }
          { Region Lers~{k} ;      Branch {NL1~{k}, NL2~{k}} ; }
        EndFor
      }
    }
  EndIf

  If(SymmetryFactor==4 && Flag_Cir_RotorCage) // Only one physical region in geo allow per branch
    { Name ElectricalCircuit ; Type Network ;
      Case Circuit1 {
        { Region Input1        ; Branch {100,101} ; }
        { Region L1            ; Branch {101,102} ; }
        { Region Stator_Ind_Ap ; Branch {102,100} ; }
      }
      Case Circuit2 {
        { Region Input2        ; Branch {200,201} ; }
        { Region L2            ; Branch {201,202} ; }
        { Region Stator_Ind_Bp ; Branch {202,200} ; }
      }
      Case Circuit3 {
        { Region Input3        ; Branch {300,301} ; }
        { Region L3            ; Branch {301,302} ; }
        { Region Stator_Ind_Cm ; Branch {300,302} ; }
      }
      Case Circuit4 {
        For k In {1:nbRotorBars}
          { Region Rotor_Bar~{k} ; Branch {NB1~{k}, NB2~{k}} ; }
          { Region Rers~{k} ;      Branch {NR1~{k}, NR2~{k}} ; }
          { Region Lers~{k} ;      Branch {NL1~{k}, NL2~{k}} ; }
        EndFor
      }
    }
  EndIf
}
