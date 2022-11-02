
radius_cylinder=40; 

$fn=100; // nbr faces

difference() {
	cylinder (h = 80, r=radius_cylinder, center = true);
	cylinder (h = 82, r=radius_cylinder-2.5, center = true);
}
