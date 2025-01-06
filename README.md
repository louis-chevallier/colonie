# colonie

Cellules : Rouge, Bleue, Vide

après une itération : en 1 endroit
$p(R) = \lambda_R \sum{R_i} - \lambda_B \sum{B_i}$

si $p_B$ = proportion de B dans les environs <br>
si $p_R =$ proportion de R dans les environs <br>
$p(B) = p_B / (p_B + p_R)$

pour rendre les choses stochastiques : remplir un tableau avec une gaussienne par cellule<br>
decision = comparaison de $p(B)$ avec la cellule correspondante<br>

retirer le tableau régulierement
