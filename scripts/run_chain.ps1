# Chạy tuần tự nhiều lượt evaluate.py, tách khỏi phiên (không bị giới hạn 2h của công cụ).
# Dùng: powershell -File scripts/run_chain.ps1 -Log results/<log>.txt -Jobs "sub1|run1;sub2|run2" [-Split sample25] [-Corpus data/corpus/gold]
# (-File không nhận mảng → các job nối bằng ';' trong MỘT chuỗi)
param(
    [string]$Log = "results/chain.log",
    [string]$Jobs,
    [string]$Split = "sample25",
    [string]$Corpus = "data/corpus/gold"
)
Set-Location (Split-Path -Parent $PSScriptRoot)
foreach ($job in ($Jobs -split ';' | Where-Object { $_.Trim() })) {
    $sub, $run = $job.Split("|")
    "[$(Get-Date -Format s)] START $sub -> $run" | Out-File -Append -Encoding utf8 $Log
    & .\.venv\Scripts\python.exe eval/evaluate.py --submission $sub --split $Split --corpus $Corpus --run $run --quant 4bit --offline --overwrite *>> $Log
    "[$(Get-Date -Format s)] END $sub rc=$LASTEXITCODE" | Out-File -Append -Encoding utf8 $Log
}
"[$(Get-Date -Format s)] CHAIN DONE" | Out-File -Append -Encoding utf8 $Log
