radius_cylinder=40;
radius_center_hole = 13;

radius_path=25;
num_holes=8;
radius_holes=1.5;

$fn=100; // nbr faces

difference(){
    difference() {
        cylinder (h = 3, r=radius_cylinder, center = true);
        cylinder (h = 6, r=radius_center_hole, center = true);
    }
    for (i=[1:num_holes])  {
        translate([radius_path*cos(i*(360/num_holes)),radius_path*sin(i*(360/num_holes)),0]) 
        cylinder(r=radius_holes,h=10, center = true);
    }
}