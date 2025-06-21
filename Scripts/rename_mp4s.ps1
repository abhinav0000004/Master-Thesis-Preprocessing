# rename all (.mp4) video files in folder
param (
    [Parameter(Mandatory = $true)]
    [int]$subjectId
)
$subject = "s$subjectId"
Get-ChildItem * | Rename-Item -NewName { $_.Name -replace '^\d{8}_','' } # remove recording date
Get-ChildItem * | Rename-Item -NewName { $_.Name -replace '^[A-Za-z]+', $subject } # participant name to subject id 
Get-ChildItem * | Rename-Item -NewName { $_.Name -replace '_\d', '' } # remove exercise iteration (_1 or _2)
Get-ChildItem * | Rename-Item -NewName { $_.Name -replace '\.\d{14}(?=\.[^.]+$)', '' } # remove isodatetime from video files
Get-ChildItem * | Rename-Item -NewName { $_.Name -replace '.59487233', '_c1' } # rename camera 1
Get-ChildItem * | Rename-Item -NewName { $_.Name -replace '.57990848', '_c2' } # rename camera 2

Get-ChildItem * | Rename-Item -NewName { $_.Name -replace '_jumpingjack_', '_jumpingjacks_' } # some variations were misspelled as 'jumpingjack' instead of 'jumpingjacks'
Get-ChildItem * | Rename-Item -NewName { $_.Name -replace '_converation_', '_conversation_' } # some variations were misspelled as 'converation' instead of 'conversation'
Get-ChildItem * | Rename-Item -NewName { $_.Name -replace '_human_', '_person_' } # some variations were misspelled as 'human' instead of 'person'
Get-ChildItem * | Rename-Item -NewName { $_.Name -replace '_objects', '_object' } # some variations were misspelled as 'objects' instead of 'object'
