// May 2013 - Authors: J. Gyselinck, R.V. Sabariego

Solver.AutoShowLastStep = 1;
Mesh.Algorithm = 1;

Include "im_3kW_data.geo" ;

Include "im_3kW_rotor.geo" ;
Include "im_3kW_stator.geo" ;


// For nice visualisation...
//Mesh.Light = 0 ;

Hide { Point{ Point '*' }; }
Hide { Line{ Line '*' }; }
Show { Line{ nicepos_rotor[], nicepos_stator[] }; }

Physical Line(NICEPOS) = { nicepos_rotor[], nicepos_stator[] };

//For post-processing...
//View[PostProcessing.NbViews-1].Light = 0;
View[PostProcessing.NbViews-1].NbIso = 25; // Number of intervals
View[PostProcessing.NbViews-1].IntervalsType = 1;
