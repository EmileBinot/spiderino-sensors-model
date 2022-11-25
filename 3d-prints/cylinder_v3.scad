
radius_cylinder=40; 
radius_cylinder2=radius_cylinder+5; 

$fn=100; // nbr faces

radius_cylinder=40;
radius_center_hole = 16;

radius_path=25;
num_holes=8;
radius_holes=1.5;


difference() {

    

difference() {
	cylinder (h = 60-3-3, r=radius_cylinder, center = true);
	cylinder (h = 82, r=radius_cylinder-2.5, center = true);
    
}

translate([0,0,-28.5]){
$fn=100; // nbr faces
rotate([0, 180, 0]) {
union(){
    difference(){
        difference() {
            cylinder (h = 3, r=radius_cylinder, center = true);
            cylinder (h = 6, r=radius_center_hole, center = true);
        }
        union(){
        for (i=[1:num_holes])  {
            translate([radius_path*cos(i*(360/num_holes)+360/(num_holes*2)),radius_path*sin(i*(360/num_holes)+360/(num_holes*2)),0]){
            cylinder(r=radius_holes,h=10, center = true);}
        };
        translate([radius_path+6,0,0]) {cylinder(r=radius_holes,h=10, center = true);};
        }
    }
    difference(){
        union(){cylinder (h = 3, r = 7, center = true);
        cube([33,5,3], center = true);}
        cylinder(r=radius_holes,h=10, center = true);
    }

    for (i=[1:4])  {
    translate([(radius_cylinder-2)*cos(i*(360/4)+360/4),(radius_cylinder-2)*sin(i*(360/4)+360/4),-3]){
    cube([4,4,4], center = true);}
    };
}
}
}
}