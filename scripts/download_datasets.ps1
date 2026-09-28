param(
    [string]$Destination = (Join-Path (Split-Path -Parent $PSScriptRoot) "datasets")
)

$ErrorActionPreference = "Stop"

$destinationPath = [System.IO.Path]::GetFullPath($Destination)
$archivePath = Join-Path $destinationPath "_archives"
New-Item -ItemType Directory -Force -Path $destinationPath, $archivePath | Out-Null

function Get-DatasetArchive {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [string]$Url,
        [Parameter(Mandatory)] [string]$FileName,
        [string]$Md5,
        [string]$Sha256
    )

    $target = Join-Path $archivePath $FileName
    $expectedHash = if ($Sha256) { $Sha256 } else { $Md5 }
    $hashAlgorithm = if ($Sha256) { "SHA256" } else { "MD5" }
    if (Test-Path -LiteralPath $target) {
        if ($expectedHash) {
            $actual = (Get-FileHash -LiteralPath $target -Algorithm $hashAlgorithm).Hash.ToLowerInvariant()
            if ($actual -eq $expectedHash.ToLowerInvariant()) {
                Write-Host "[$Name] Archive already downloaded and verified."
                return $target
            }
        } else {
            Write-Host "[$Name] Resuming or reusing existing archive."
        }
    }

    Write-Host "[$Name] Downloading $Url"
    & curl.exe --fail --location --retry 5 --retry-delay 3 --retry-all-errors --continue-at - --output $target $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed for $Name (curl exit code $LASTEXITCODE)."
    }

    if ($expectedHash) {
        $actual = (Get-FileHash -LiteralPath $target -Algorithm $hashAlgorithm).Hash.ToLowerInvariant()
        if ($actual -ne $expectedHash.ToLowerInvariant()) {
            throw "Checksum mismatch for $Name. Expected $expectedHash, got $actual."
        }
        Write-Host "[$Name] $hashAlgorithm verified."
    }

    return $target
}

function Expand-DatasetArchive {
    param(
        [Parameter(Mandatory)] [string]$Name,
        [Parameter(Mandatory)] [string]$Archive,
        [Parameter(Mandatory)] [string]$ExtractTo,
        [Parameter(Mandatory)] [string]$CompletionMarker
    )

    $marker = Join-Path $ExtractTo $CompletionMarker
    if (Test-Path -LiteralPath $marker) {
        Write-Host "[$Name] Already extracted."
        return
    }

    New-Item -ItemType Directory -Force -Path $ExtractTo | Out-Null
    Write-Host "[$Name] Extracting to $ExtractTo"
    & tar.exe -xf $Archive -C $ExtractTo
    if ($LASTEXITCODE -ne 0) {
        throw "Extraction failed for $Name (tar exit code $LASTEXITCODE)."
    }
    if (-not (Test-Path -LiteralPath $marker)) {
        throw "Extraction marker was not created for $Name`: $marker"
    }
}

$dtd = Get-DatasetArchive `
    -Name "DTD" `
    -Url "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz" `
    -FileName "dtd-r1.0.1.tar.gz" `
    -Md5 "fff73e5086ae6bdbea199a49dfb8a4c1"
Expand-DatasetArchive -Name "DTD" -Archive $dtd -ExtractTo $destinationPath -CompletionMarker "dtd\images"

$aircraft = Get-DatasetArchive `
    -Name "FGVC Aircraft" `
    -Url "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz" `
    -FileName "fgvc-aircraft-2013b.tar.gz" `
    -Md5 "d4acdd33327262359767eeaa97a4f732"
Expand-DatasetArchive -Name "FGVC Aircraft" -Archive $aircraft -ExtractTo $destinationPath -CompletionMarker "fgvc-aircraft-2013b\data\images"

$cub = Get-DatasetArchive `
    -Name "CUB-200-2011" `
    -Url "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1" `
    -FileName "CUB_200_2011.tgz" `
    -Md5 "97eceeb196236b17998738112f37df78"
Expand-DatasetArchive -Name "CUB-200-2011" -Archive $cub -ExtractTo $destinationPath -CompletionMarker "CUB_200_2011\images"

$dogsPath = Join-Path $destinationPath "stanford_dogs"
$dogImages = Get-DatasetArchive `
    -Name "Stanford Dogs images" `
    -Url "https://huggingface.co/datasets/dgrnd4/stanford_dog_dataset/resolve/main/stanford_dog_dataset.zip?download=true" `
    -FileName "stanford-dogs-images-mirror.zip" `
    -Sha256 "d7e7c4d08d0df25964f9a2835e0f91fcfafd3d26c6ad083cd82d7cece4be42b4"
if (-not (Test-Path -LiteralPath (Join-Path $dogsPath "Images"))) {
    New-Item -ItemType Directory -Force -Path $dogsPath | Out-Null
    Write-Host "[Stanford Dogs images] Extracting mirror archive."
    & tar.exe -xf $dogImages -C $dogsPath stanford_dog_dataset
    if ($LASTEXITCODE -ne 0) {
        throw "Extraction failed for Stanford Dogs images (tar exit code $LASTEXITCODE)."
    }
    Move-Item -LiteralPath (Join-Path $dogsPath "stanford_dog_dataset") -Destination (Join-Path $dogsPath "Images")
}

$dogMetadataPath = Join-Path $dogsPath "metadata.csv"
if (-not (Test-Path -LiteralPath $dogMetadataPath)) {
    $dogMetadata = Get-DatasetArchive `
        -Name "Stanford Dogs labels and bounding boxes" `
        -Url "https://huggingface.co/datasets/Alanox/stanford-dogs/resolve/main/metadata.csv" `
        -FileName "stanford-dogs-metadata.csv" `
        -Sha256 "788ba75b66065a990ff7dd9fd1525cd153e2e94ee76d2ea32dc651b084d60297"
    Copy-Item -LiteralPath $dogMetadata -Destination $dogMetadataPath
}

$dogLists = Get-DatasetArchive `
    -Name "Stanford Dogs splits" `
    -Url "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar" `
    -FileName "stanford-dogs-lists.tar" `
    -Sha256 "34b47cacd9a98b5d150e084f24d29391c084c55272295ec65c85651bc35f4d6c"
Expand-DatasetArchive -Name "Stanford Dogs splits" -Archive $dogLists -ExtractTo $dogsPath -CompletionMarker "train_list.mat"

$petsPath = Join-Path $destinationPath "oxford_iiit_pet"
$petImages = Get-DatasetArchive `
    -Name "Oxford-IIIT Pet images" `
    -Url "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz" `
    -FileName "oxford-pets-images.tar.gz" `
    -Md5 "5c4f3ee8e5d25df40f4fd59a7f44e54c"
Expand-DatasetArchive -Name "Oxford-IIIT Pet images" -Archive $petImages -ExtractTo $petsPath -CompletionMarker "images"

$petAnnotations = Get-DatasetArchive `
    -Name "Oxford-IIIT Pet annotations" `
    -Url "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz" `
    -FileName "oxford-pets-annotations.tar.gz" `
    -Md5 "95a8c909bbe2e81eed6a22bccdf3f68f"
Expand-DatasetArchive -Name "Oxford-IIIT Pet annotations" -Archive $petAnnotations -ExtractTo $petsPath -CompletionMarker "annotations\trainval.txt"

Write-Host "All requested datasets are downloaded and extracted under $destinationPath"
