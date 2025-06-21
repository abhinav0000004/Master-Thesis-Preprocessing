# Extract all frames of all videos in folder into a seperate image directory with one folder containing all images
# Uses ffmpeg
param (
    [Parameter(Mandatory = $true)]
    [string]$videoDir,

    [Parameter(Mandatory = $true)]
    [string]$imageDir
)

# Validate that the video directory exists
if (-Not (Test-Path -Path $videoDir)) {
    Write-Error "The video directory '$videoDir' does not exist."
    exit 1
}

# Create the images directory if it doesn't exist
if (!(Test-Path -Path $imageDir)) {
    New-Item -ItemType Directory -Path $imageDir
}

# Loop through each video file in the video directory
Get-ChildItem -Path $videoDir -Filter *.mp4 | ForEach-Object {
    $videoPath = $_.FullName
    $videoName = $_.BaseName

    # Create a subdirectory in images for each video
    $outputDir = "$imageDir\$videoName"
    if (!(Test-Path -Path $outputDir)) {
        New-Item -ItemType Directory -Path $outputDir
    }

    # Run FFmpeg to extract frames from the video
    # -i specifies the input video file
    # -q:v 2 specifies the JPEG quality level (2 is high quality, lower number means better quality)
    # %04d.jpg names files with 4-digit numbering (0001.jpg, 0002.jpg, etc.)
    ffmpeg -i $videoPath -q:v 2 -start_number 0 "$outputDir\%04d.jpg"
}

Write-Output "Frame extraction completed for all videos."