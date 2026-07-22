class Spkreg < Formula
  desc "Recognise returning speakers across whispermlx diarizations"
  homepage "https://github.com/mainpart/auto-speakers"
  url "https://github.com/mainpart/auto-speakers/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "01a495073e0a1ddf607b2f69f16dedc9fa808db92105068d280bb6aaa9ad9162"
  license "MIT"

  depends_on "python@3.12"

  def install
    bin.install "spkreg.py" => "spkreg"
  end

  test do
    assert_match "spkreg", shell_output("#{bin}/spkreg --help")
  end
end
