
radius_cylinder=40; 
radius_cylinder2=radius_cylinder+5; 

$fn=100; // nbr faces

difference() {
	cylinder (h = 40, r=radius_cylinder, center = true);
	cylinder (h = 82, r=radius_cylinder-2.5, center = true);
    
}
