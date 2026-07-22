class Spkreg < Formula
  desc "Recognise returning speakers across whispermlx diarizations"
  homepage "https://github.com/mainpart/auto-speakers"
  license "MIT"
  head "https://github.com/mainpart/auto-speakers.git", branch: "main"

  depends_on "python@3.12"

  def install
    bin.install "spkreg.py" => "spkreg"
  end

  test do
    assert_match "spkreg", shell_output("#{bin}/spkreg --help")
  end
end
