/**
 * Created on Sep 11, 2011
 */
package info.halo9pan.examples.prospring3.ch4;

/**
 * @author Clarence
 *
 */
public interface MessageRenderer {

	public void render();

	public void setMessageProvider(MessageProvider provider);

	public MessageProvider getMessageProvider();

}
