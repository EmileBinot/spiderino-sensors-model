//Slab
//cube(size = [74.5,74.5,3]);
batteryl=27;
batteryL=46;
batteryh=13;
ep = 2;

//Tray
//top
topx =0;
topy = 0;
topz =0;
translate([topx,topy,topz]);

cube(size = [batteryl+ep,batteryL+ep*2,ep]);

//side
translate([topx,topy,topz+ep])cube(size = [ep,batteryL+ep*2,batteryh]);
//translate([topx+batteryl+ep,topy,topz+ep])cube(size = [ep,batteryL+ep,batteryh]);
//cube(size = [2,batteryL,batteryh]);


translate([topx+ep,topy,topz+ep])cube(size = [batteryl,ep,batteryh]);
translate([topx+ep,topy+batteryL+ep,topz+ep])cube(size = [batteryl,ep,batteryh]);
//cube(size = [13,5,2]);






