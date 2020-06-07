// May 2013 - Authors: J. Gyselinck, R.V. Sabariego

// 3kW induction machine from Johan Gyselinck's PhD
/*
Some articles where it has been used:
J. Gyselinck, L. Vandevelde, and J. Melkebeek,
"Multi-slice modeling of electrical machines with skewed slots - The skew
discretization error,” IEEE Trans. Magn., vol. 37, pp. 3233–3237, Sept. 2002.

J. Gyselinck, L. Vandevelde, P. Dular, C. Geuzaine, W. Legros,
"A General Method for the Frequency Domain FE Modeling of Rotating
Electromagnetic Devices", IEEE Trans. Magn., Vol. 39, No. 3, May 2003.
*/

u = 1e-3 ; // unit = mm
deg2rad = Pi/180 ;

pp = "Input/Constructive parameters/";

DefineConstant[
  NbrPolesInModel = { 1, Choices{ 1 = "1", 2 = "2", 4 = "4" },
    Name "Input/20Number of poles in FE model", Highlight "Blue"},
  InitialRotorAngle_deg = { 10, Name "Input/20Initial rotor angle [deg]",
    Highlight "AliceBlue"},
  Flag_OpenRotor = {1, Choices{0,1}, Name "Input/39Open slots in rotor"}
];

NbrPolesTot = 4; // number of poles in complete cross-section

SymmetryFactor = NbrPolesTot/NbrPolesInModel;
Flag_Symmetry = (SymmetryFactor==1)?0:1;

// Rotor
NbrSectTot = 32; // number of rotor teeth
NbrSect = NbrSectTot*NbrPolesInModel/NbrPolesTot; // number of rotor teeth in FE model

//Stator
NbrSectStatorTot = 36; // number of stator teeth
NbrSectStator = NbrSectStatorTot*NbrPolesInModel/NbrPolesTot; // number of stator teeth in FE model

//--------------------------------------------------------------------------------

InitialRotorAngle = InitialRotorAngle_deg*deg2rad ; // initial rotor angle, 0 if aligned

//-----------------------------------------------------------------------
Freq = 50 ;

DefineConstant[
  AG = {u*0.47, Name StrCat[pp, "Airgap width [m]"], Closed 1}
];

Lz = u*127;
AxialLength = Lz ;

VA = 220  ; // 220 V \Delta or 380 V Y of supply voltage
IA = 12.3 ; // 12.3 A \Delta or 6.7 A Y

// Stator and rotor made of VH800-65D, laminated block, fill factor 96%

sigma_fe   = 0 ;

//----------------------------------------------------------
// value of sigma with homogenization of order zero
// sigma_fe = {12.6e6, Label "Stator/rotor equivalent conductivity [S/m]", Path Str[pp]},

// it only makes sense when adding a term in the formulation: d^2/12*sigma*d_t b
// so that we can account for losses in the laminationed domain, which is still non-conducting
// not accessible thus via the Onelab GUI
//----------------------------------------------------------

DefineConstant[
  sigma_bars = {26.7e6,
    Name StrCat[pp, "ss"], Label "Conductivity of rotor bars [S/m]"}, // AlSi
  mur_fe = {1500,
    Name StrCat[pp, "Relative permeability for linear case"]}
];

dIron = 0.65e-3; // thickness electrical steel, in m

//R_endring_segment = 1.702; // resistance of two endring segments in series in ohm (JG's PhD)
//L_endring_segment = 6.62e-9; // inductance of two endring segments in series in H

//FIXME: I think those parameters shouldn't be changed from interface, as they are calculated analytically
DefineConstant[
  R_endring_segment = {0.836e-6,
    Name StrCat[pp, "Resistance of two endring segments in series [Ohm]"]}, // (JG's PhD)
  L_endring_segment = {4.8e-9,
    Name StrCat[pp, "Inductance of two endring segments in series [H]"]},
  Rs = {2.2,
    Name StrCat[pp, "Resistance per stator phase [Ohm]"]},
  Ls = {0.87e-3,
    Name StrCat[pp, "Endwinding inductance per stator phase [H]"]}
];

Xs = 2*Pi*Freq * Ls ;

DefineConstant[
  Ns ={ 6*34,
    Name StrCat[pp, "Total number of turns in series per phase"]}
] ;

rpm_nominal = 1420 ; // turns/min
rpm_other = 1430;

inertia_fe = 5.63*1e-3 ; //kg*m^2

//-----------------------------------------------------------------------
// Physical numbers
//-----------------------------------------------------------------------
// Numbers for physical regions in .geo and .pro files
// ----------------------------------------------------
// Rotor
ROTOR_FE     = 20000 ;
ROTOR_SHAFT  = 20001 ;
ROTOR_SLOTOPENING = 20002 ; // RotorSlotOpening
ROTOR_AIRGAP      = 20003 ; // RotorAirgapLayer

ROTOR_BAR = 30000 ;
ROTOR_BAR01 = ROTOR_BAR+1;  ROTOR_BAR11 = ROTOR_BAR+11;  ROTOR_BAR21 = ROTOR_BAR+21;  ROTOR_BAR31 = ROTOR_BAR+31;
ROTOR_BAR02 = ROTOR_BAR+2;  ROTOR_BAR12 = ROTOR_BAR+12;  ROTOR_BAR22 = ROTOR_BAR+22;  ROTOR_BAR32 = ROTOR_BAR+32;
ROTOR_BAR03 = ROTOR_BAR+3;  ROTOR_BAR13 = ROTOR_BAR+13;  ROTOR_BAR23 = ROTOR_BAR+23;  ROTOR_BAR33 = ROTOR_BAR+33;
ROTOR_BAR04 = ROTOR_BAR+4;  ROTOR_BAR14 = ROTOR_BAR+14;  ROTOR_BAR24 = ROTOR_BAR+24;  ROTOR_BAR34 = ROTOR_BAR+34;
ROTOR_BAR05 = ROTOR_BAR+5;  ROTOR_BAR15 = ROTOR_BAR+15;  ROTOR_BAR25 = ROTOR_BAR+25;  ROTOR_BAR35 = ROTOR_BAR+35;
ROTOR_BAR06 = ROTOR_BAR+6;  ROTOR_BAR16 = ROTOR_BAR+16;  ROTOR_BAR26 = ROTOR_BAR+26;  ROTOR_BAR36 = ROTOR_BAR+36;
ROTOR_BAR07 = ROTOR_BAR+7;  ROTOR_BAR17 = ROTOR_BAR+17;  ROTOR_BAR27 = ROTOR_BAR+27;  ROTOR_BAR37 = ROTOR_BAR+37;
ROTOR_BAR08 = ROTOR_BAR+8;  ROTOR_BAR18 = ROTOR_BAR+18;  ROTOR_BAR28 = ROTOR_BAR+28;  ROTOR_BAR38 = ROTOR_BAR+38;
ROTOR_BAR09 = ROTOR_BAR+9;  ROTOR_BAR19 = ROTOR_BAR+19;  ROTOR_BAR29 = ROTOR_BAR+29;  ROTOR_BAR39 = ROTOR_BAR+39;
ROTOR_BAR10 = ROTOR_BAR+10; ROTOR_BAR20 = ROTOR_BAR+20;  ROTOR_BAR30 = ROTOR_BAR+30;  ROTOR_BAR40 = ROTOR_BAR+40;

ROTOR_BND_MOVING_BAND = 22000 ; // Index for first line (1/8 model->1; full model->8)
MB_R1 = ROTOR_BND_MOVING_BAND+0 ;
MB_R2 = ROTOR_BND_MOVING_BAND+1 ;
MB_R3 = ROTOR_BND_MOVING_BAND+2 ;
MB_R4 = ROTOR_BND_MOVING_BAND+3 ;

ROTOR_BND_A0 = 21000 ; // RotorPeriod_Reference
ROTOR_BND_A1 = 21001 ; // RotorPeriod_Dependent

SURF_INT     = 21002 ; // OuterShaft

// Stator
STATOR_FE          = 10000 ;
STATOR_SLOTOPENING = 11000 ; // Slot opening
STATOR_AIRGAP      = 12000 ; // Between the moving band and the slot opening

STATOR_IND = 13000;
STATOR_IND_AP = STATOR_IND + 1; STATOR_IND_CM = STATOR_IND + 2; STATOR_IND_BP = STATOR_IND + 3 ;
STATOR_IND_AM = STATOR_IND + 4; STATOR_IND_CP = STATOR_IND + 5; STATOR_IND_BM = STATOR_IND + 6 ;

STATOR_BND_MOVING_BAND = 14000 ;// Index for first line (1/8 model->1; full model->8)
MB_S1 = STATOR_BND_MOVING_BAND+0 ;
MB_S2 = STATOR_BND_MOVING_BAND+1 ;
MB_S3 = STATOR_BND_MOVING_BAND+2 ;
MB_S4 = STATOR_BND_MOVING_BAND+3 ;

STATOR_BND_A0          = 15000 ;
STATOR_BND_A1          = 15001 ;

SURF_EXT = 16000 ; // outer boundary

MOVING_BAND = 9999 ;

NICEPOS = 111111 ;
