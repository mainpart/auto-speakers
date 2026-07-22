class Spkreg < Formula
  desc "Recognise returning speakers across whispermlx diarizations"
  homepage "https://github.com/mainpart/auto-speakers"
  url "https://github.com/mainpart/auto-speakers/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "REPLACE_WITH_TARBALL_SHA256"
  license "MIT"

  depends_on "python@3.12"

  def install
    bin.install "spkreg.py" => "spkreg"
  end

  test do
    assert_match "spkreg", shell_output("#{bin}/spkreg --help")
  end
end
