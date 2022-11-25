radius_cylinder=40; 
radius_cylinder2=radius_cylinder+6; 

$fn=100; 



difference() {
	cylinder (h = 18, r=radius_cylinder2, center = true);
    cylinder (h = 82, r=radius_cylinder2-2.5, center = true);
}

translate([0,0,-9])
difference() {
	cylinder (h = 2.5, r=radius_cylinder2, center = true);
    cylinder (h = 2.6, r=radius_cylinder, center = true);
}
