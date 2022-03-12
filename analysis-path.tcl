proc flux {start midpoint name} {

    mol load pdb $name.pdb dcd $name.dcd

    set x1 [expr $midpoint]
    set x2 [expr $midpoint+2]
    
    set outfile [open flux_outcome_$name.csv w]
    
    for {set f $start } { $f < [molinfo top get numframes]} {incr f} {
        set sel1 [atomselect top "name RC and x < $x2 and x > $x1" frame $f]
        set sel2 [atomselect top "name RC and x > $x1" frame $f]
        set num1 [$sel1 num]
        set num2 [$sel2 num]
        set val [expr $num1/40/100]
        # 40 something and 100 steps 
        puts $outfile "$f, $num1, $num2, $val"
        
        unset sel1
        unset sel2
        unset num1
        unset num2
        unset val
    }
    close $outfile
}

proc flux2 {start name} {

    mol load pdb $name.pdb dcd $name.dcd
   
    set outfile [open xcoord_$name.csv w]
    set totalFrame [molinfo top get numframes]
    set maxFrame [expr $totalFrame - 1]
    
    for {set f $start} {$f < $totalFrame} {incr f} {
        set sel1 [atomselect top "name RC" frame $f]
        set coord [$sel1 get {x}]
        puts $outfile "$f $coord"
    }
   
    close $outfile
}

proc fluxy {start name} {

    mol load pdb $name.pdb dcd $name.dcd
   
    set outfile [open ycoord_$name.csv w]
    set totalFrame [molinfo top get numframes]
    set maxFrame [expr $totalFrame - 1]
    
    for {set f $start} {$f < $totalFrame} {incr f} {
        set sel1 [atomselect top "name RC" frame $f]
        set coord [$sel1 get {y}]
        puts $outfile "$f $coord"
    }
   
    close $outfile
}

proc flux3 {start ref1 numSim repeat} {
    for {set n $ref1} {$n <= $numSim} {incr n} {
        for {set m 0} {$m <= $repeat} {incr m} {
            set testNum $n
            set repNum $m
            flux2 $start test$n-$m
        }
    }
}

proc rmsd {start name} {
    
    mol load pdb $name.pdb dcd $name.dcd
    
    set outfile1 [open rmsd_$name.csv w]
    set outfile2 [open msd_$name.csv w]
    set totalFrame [molinfo top get numframes]
    set maxFrame [expr $totalFrame - 1]
    
    set sel1 [atomselect top "name RC" frame $start]
    set index1 [ lsort -integer -index 0 -increasing -unique [$sel1 get {index}]]
    
    for {set f $start} {$f < $totalFrame} {incr f 1} {
                                
        set var1 $f
        set var2 $f
        foreach index $index1 {
            set ref2 [atomselect top "index $index" frame $start]
            set sel2 [atomselect top "index $index" frame $f]
            set rmsd [measure rmsd $sel2 $ref2]
            set msd [expr $rmsd*$rmsd]
            append var1 ", " "$rmsd" 
            append var2 ", " "$msd"
            
            unset ref2 
            unset sel2 
            unset rmsd
            unset msd 
            
            }
        puts $outfile1 "$var1"
        puts $outfile2 "$var2"
        
        unset var1
        unset var2
        
        }
    close $outfile1
    close $outfile2
}

proc rmsdseries {start startNo endNo} {
    for {set no $startNo} {$no < $endNo} {incr no 1} {
        rmsd $start test$no
    }
}